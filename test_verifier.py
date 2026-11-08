"""Mini Codex — Phase 4: Automated Test Generation & Verification Loops.

A LangGraph workflow that:
  1. Inspects the target module (e.g. calculator.py).
  2. Generates targeted pytest unit tests targeting both regular behavior and edge/bug cases.
  3. Executes pytest strictly in the sample_project sandbox.
  4. Intercepts failures, analyzes the pytest traceback with Gemini, and refines the patch.
  5. Retries automatically (up to max_retries) until all tests pass green.
  6. Never claims a bug is fixed unless tests genuinely pass!
"""

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict
from dotenv import load_dotenv

from google import genai
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from tools import (
    get_workspace_dir,
    delete_backup,
    list_python_files,
    read_python_file,
    create_backup,
    apply_patch,
    restore_backup,
    write_test_file,
    run_pytest,
)

# 1. Terminal configuration for Windows PowerShell (UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] GEMINI_API_KEY is missing from .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"


# =====================================================================
# Pydantic Schemas for Structured Model Responses
# =====================================================================
class GeneratedTestSuite(BaseModel):
    """Schema for test generation output."""

    test_code: str = Field(
        description="The complete, runnable pytest Python test suite. Must import from the target module."
    )
    test_cases_summary: list[str] = Field(
        description="List of specific test scenarios covered (e.g., normal math, edge cases, zero division)."
    )


class RefinedPatch(BaseModel):
    """Schema for test failure reflection and patch revision."""

    thought_process: str = Field(
        description="Analysis of why the tests failed based on the pytest error traceback."
    )
    fixed_code: str = Field(
        description="The complete revised Python source code for the target module with bugs fixed."
    )
    changes_made: list[str] = Field(
        description="Brief list of specific code adjustments made in this revision."
    )


# =====================================================================
# LangGraph State Definition
# =====================================================================
class TestVerificationState(TypedDict):
    """State passed across the Phase 4 test verification graph."""

    filename: str
    original_code: str
    current_code: str
    test_filename: str
    test_code: str
    test_output: str
    tests_passed: bool
    retry_count: int
    max_retries: int
    failure_analysis: str
    final_status: str


# =====================================================================
# Graph Nodes
# =====================================================================
def read_target_code(state: TestVerificationState) -> dict:
    """Node 1: Read the target file and create an undo backup."""
    filename = state["filename"]
    print(f"\n[Node 1: read_target_code] Reading '{filename}' and creating backup...")

    code = read_python_file(filename)
    if code.startswith("Access Denied") or code.startswith("Error"):
        raise ValueError(f"Could not read target file: {code}")

    bak_msg = create_backup(filename)
    print(f"  -> {bak_msg}")

    # Deduce corresponding test file name (e.g., calculator.py -> test_calculator.py, utils/helpers.py -> utils/test_helpers.py)
    p = Path(filename.replace("\\", "/"))
    if p.parent != Path("."):
        test_filename = f"{p.parent}/test_{p.stem}.py".replace("\\", "/")
    else:
        test_filename = f"test_{p.stem}.py"

    return {
        "original_code": code,
        "current_code": code,
        "test_filename": test_filename,
    }


def _clean_code_fence(code: str) -> str:
    """Strip accidental markdown code fences (e.g. ```python) from LLM output."""
    clean = code.strip()
    if clean.startswith("```python"):
        clean = clean[9:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def generate_tests_node(state: TestVerificationState) -> dict:
    """Node 2: Generates a pytest test suite covering both happy and edge/bug paths."""
    filename = state["filename"].replace("\\", "/")
    p = Path(filename)
    module_import = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
    code = state["current_code"]

    print(f"\n[Node 2: generate_tests] Writing automated unit tests for '{filename}'...")

    prompt = f"""You are an expert QA and Software Test Engineer.
Write a comprehensive pytest unit test suite for the following Python module ('{filename}'):

```python
{code}
```

Requirements:
1. Import all functions and classes from `{module_import}` (e.g., `from {module_import} import ...` or `import {module_import}`).
2. Write unit tests testing standard expected behavior (happy paths).
3. Write edge-case and boundary tests (boundary conditions, empty/null values, unexpected types, zero/negative inputs, exception handling).
4. Tests should be clear, idiomatic pytest functions named `test_*`.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=dict(
            response_mime_type="application/json",
            response_schema=GeneratedTestSuite,
            temperature=0.0,
        ),
    )

    suite = GeneratedTestSuite.model_validate_json(response.text)
    test_filename = state["test_filename"]
    clean_test_code = _clean_code_fence(suite.test_code)

    # Write test file to sample_project
    write_res = write_test_file(test_filename, clean_test_code)
    print(f"  -> {write_res}")
    print(f"  -> Scenarios covered: {len(suite.test_cases_summary)} test cases:")
    for c in suite.test_cases_summary:
        print(f"     • {c}")

    return {
        "test_code": clean_test_code,
    }


def run_tests_node(state: TestVerificationState) -> dict:
    """Node 3: Executes pytest strictly against the generated test file."""
    test_filename = state["test_filename"]
    print(f"\n[Node 3: run_tests] Executing pytest against '{test_filename}' (Attempt {state.get('retry_count', 0) + 1})...")

    res = run_pytest(test_filename)
    passed = res["passed"]
    output = res["output"]

    if passed:
        print("  -> ✅ Result: All tests PASSED!")
    else:
        print("  -> ❌ Result: Tests FAILED. Errors detected in test run.")

    # Show brief output summary
    summary_lines = [l for l in output.splitlines() if "PASSED" in l or "FAILED" in l or "ERROR" in l]
    if summary_lines:
        print("  -> Test Summary:")
        for s in summary_lines[:6]:
            print(f"     {s}")

    return {
        "tests_passed": passed,
        "test_output": output,
    }


def analyze_and_refine_patch_node(state: TestVerificationState) -> dict:
    """Node 4: Reflects on failed tests and generates a revised patch."""
    filename = state["filename"]
    current_code = state["current_code"]
    test_output = state["test_output"]
    attempt = state.get("retry_count", 0) + 1

    print(f"\n[Node 4: analyze_and_refine_patch] Self-Correction Turn #{attempt}...")
    print("  -> Feeding pytest traceback to Gemini for diagnosis...")

    prompt = f"""You are a Principal Software Engineer debugging failing tests.
Target file: '{filename}'

Current Code:
```python
{current_code}
```

Pytest Failure Output:
```text
{test_output}
```

Instructions:
1. Carefully diagnose the exact reason each test failed from the pytest output and traceback.
2. Fix the implementation in `{filename}` to make all tests pass green.
3. Preserve existing function signatures, docstrings, and non-failing logic.
4. Return the complete, corrected source code.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=dict(
            response_mime_type="application/json",
            response_schema=RefinedPatch,
            temperature=0.0,
        ),
    )

    patch = RefinedPatch.model_validate_json(response.text)

    print(f"\n  [Diagnosis]: {patch.thought_process[:200]}...")
    print("  [Changes applied]:")
    for ch in patch.changes_made:
        print(f"    • {ch}")

    clean_fixed_code = _clean_code_fence(patch.fixed_code)

    # Apply the revised patch to the target file
    apply_msg = apply_patch(filename, clean_fixed_code)
    print(f"  -> {apply_msg}")

    return {
        "current_code": clean_fixed_code,
        "retry_count": attempt,
        "failure_analysis": patch.thought_process,
    }


def success_node(state: TestVerificationState) -> dict:
    """Node 5: Verification Confirmed — All tests passed green!"""
    filename = state["filename"]
    retries = state.get("retry_count", 0)

    print("\n" + "=" * 65)
    print("  🎉 VERIFICATION SUCCESS: All Pytest Tests Passed Green!")
    print("=" * 65)
    print(f"Target Module     : {filename}")
    print(f"Self-Corrections  : {retries} retry loop(s) needed")
    print(f"Status            : Verified Working against Unit Test Suite")
    print("=" * 65)

    # Clean up the backup since tests passed
    if delete_backup(filename):
        print(f"🧹 Cleaned up temporary backup for '{filename}'.")

    return {
        "final_status": f"SUCCESS: All tests passed after {retries} correction loop(s)."
    }


def failure_node(state: TestVerificationState) -> dict:
    """Node 6: Reached maximum retries without passing tests."""
    filename = state["filename"]
    max_retries = state.get("max_retries", 3)

    print("\n" + "=" * 65)
    print(f"  ❌ VERIFICATION FAILED: Max Retries ({max_retries}) Exhausted.")
    print("=" * 65)
    print(f"Restoring '{filename}' to its original state from backup...")

    restore_msg = restore_backup(filename)
    print(f"  -> {restore_msg}")
    print("=" * 65)

    return {
        "final_status": f"FAILED: Tests could not pass within {max_retries} attempts. Rolled back."
    }


# =====================================================================
# Conditional Edge (Router)
# =====================================================================
def evaluate_test_results(state: TestVerificationState) -> Literal["success", "refine_patch", "failure"]:
    """Determines whether to finish successfully, retry/refine, or abort."""
    if state.get("tests_passed", False):
        return "success"

    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "refine_patch"

    return "failure"


# =====================================================================
# Graph Construction & Compilation
# =====================================================================
def build_test_verifier_graph():
    """Builds the Phase 4 Test-Driven Verification StateGraph."""
    workflow = StateGraph(TestVerificationState)

    # 1. Add nodes
    workflow.add_node("read_target_code", read_target_code)
    workflow.add_node("generate_tests", generate_tests_node)
    workflow.add_node("run_tests", run_tests_node)
    workflow.add_node("analyze_and_refine_patch", analyze_and_refine_patch_node)
    workflow.add_node("success", success_node)
    workflow.add_node("failure", failure_node)

    # 2. Add fixed edges
    workflow.add_edge(START, "read_target_code")
    workflow.add_edge("read_target_code", "generate_tests")
    workflow.add_edge("generate_tests", "run_tests")

    # 3. Add conditional routing based on pytest results
    workflow.add_conditional_edges(
        "run_tests",
        evaluate_test_results,
        {
            "success": "success",
            "refine_patch": "analyze_and_refine_patch",
            "failure": "failure",
        },
    )

    # 4. Feedback loop: after refining the patch, re-run tests!
    workflow.add_edge("analyze_and_refine_patch", "run_tests")

    # 5. Terminal edges
    workflow.add_edge("success", END)
    workflow.add_edge("failure", END)

    return workflow.compile()


verifier_graph = build_test_verifier_graph()


# =====================================================================
# CLI Runner with Persistent Menu & Error Recovery
# =====================================================================
def select_file(files: list[str]) -> str | None:
    while True:
        print("\nAvailable files for verification in sample_project:")
        for idx, f in enumerate(files, 1):
            print(f"  {idx}. {f}")
        print("  (Type 'exit' or 'q' to quit)")

        try:
            choice = input("\nEnter file number or name (default: 1. calculator.py): ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        if not choice:
            return "calculator.py"
        if choice.lower() in ("exit", "quit", "q"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        if choice in files:
            return choice
        if f"{choice}.py" in files:
            return f"{choice}.py"

        print(f"\n⚠️  [INVALID SELECTION] '{choice}' is not in sample_project.")


def main():
    print("=" * 65)
    print("  Mini Codex — Test Verification & Self-Correction (Phase 4)")
    print("=" * 65)

    files = [f for f in list_python_files() if not f.startswith("test_")]
    if not files:
        print("[ERROR] No target Python files found in sample_project!")
        return

    # Check CLI argument
    if len(sys.argv) > 1 and sys.argv[1] in files:
        target = sys.argv[1]
    else:
        target = select_file(files)
        if not target:
            print("\nGoodbye!")
            return

    initial_state = {
        "filename": target,
        "original_code": "",
        "current_code": "",
        "test_filename": "",
        "test_code": "",
        "test_output": "",
        "tests_passed": False,
        "retry_count": 0,
        "max_retries": 3,
        "failure_analysis": "",
        "final_status": "",
    }

    try:
        result = verifier_graph.invoke(initial_state)
        print(f"\nResult: {result['final_status']}")
    except Exception as e:
        print(f"\n❌ [ERROR] An error occurred during test verification: {e}")


if __name__ == "__main__":
    main()

