"""Mini Codex — Phase 3: Automated Bug Analysis & Patching.

A LangGraph workflow that:
  1. Inspects a target Python file in sample_project.
  2. Categorizes issues into Confirmed Bugs vs Suspected Smells.
  3. Generates a non-destructive proposed patch as a unified diff.
  4. Enforces a Human-in-the-Loop (HITL) approval gate.
  5. Backs up the original file (.bak) and applies the patch ONLY upon approval.
"""

import difflib
import os
import sys
import time
from typing import Literal, TypedDict
from dotenv import load_dotenv

from google import genai
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from tools import (
    list_python_files,
    read_python_file,
    create_backup,
    apply_patch,
    restore_backup,
)

# 1. Terminal configuration for Windows PowerShell
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


# =====================================================================
# Pydantic Schema for Structured Bug Analysis
# =====================================================================
class BugReport(BaseModel):
    """Structured analysis output from Gemini."""

    confirmed_bugs: list[str] = Field(
        description="Definite runtime errors, logic bugs, or crashes (e.g. inverted logic, unhandled exceptions)."
    )
    suspected_issues: list[str] = Field(
        description="Code smells, missing type hints, missing docstrings, or edge cases that might fail."
    )
    summary: str = Field(
        description="A concise 1-2 sentence overview of the code quality and findings."
    )
    fixed_code: str = Field(
        description="The complete, fixed Python code with all confirmed bugs resolved."
    )


# =====================================================================
# LangGraph State Definition
# =====================================================================
class BugAnalysisState(TypedDict):
    """The state passed through the Phase 3 graph."""

    filename: str
    original_code: str
    confirmed_bugs: list[str]
    suspected_issues: list[str]
    summary: str
    fixed_code: str
    unified_diff: str
    approved: bool
    status_message: str


# =====================================================================
# Graph Nodes
# =====================================================================
def read_target_file(state: BugAnalysisState) -> dict:
    """Node 1: Safely reads the original code of the chosen file."""
    filename = state["filename"]
    print(f"\n[Node 1: read_target_file] Reading '{filename}' from sample_project...")
    code = read_python_file(filename)

    if code.startswith("Access Denied") or code.startswith("Error"):
        raise ValueError(f"Could not read target file: {code}")

    return {"original_code": code}


def detect_bugs_node(state: BugAnalysisState) -> dict:
    """Node 2: Prompts Gemini to analyze the code and produce a structured bug report."""
    filename = state["filename"]
    code = state["original_code"]
    print(f"\n[Node 2: detect_bugs] Analyzing '{filename}' for logic errors and bugs...")

    prompt = f"""You are a Principal Software Engineer and Code Reviewer.
Analyze the following Python file ('{filename}') for bugs:

```python
{code}
```

Instructions:
1. Identify any CONFIRMED BUGS: logic errors, runtime crashes, wrong operations, or inverted arguments.
2. Identify any SUSPECTED ISSUES: missing boundary checks, unhandled edge cases, or code smells.
3. Provide a summary of your findings.
4. Provide the complete FIXED Python source code with all confirmed bugs fixed. Preserve the original function names and docstrings.
"""

    # Use gemini-2.5-flash as specified by user
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=dict(
            response_mime_type="application/json",
            response_schema=BugReport,
            temperature=0.0,
        ),
    )

    report = BugReport.model_validate_json(response.text)
    return {
        "confirmed_bugs": report.confirmed_bugs,
        "suspected_issues": report.suspected_issues,
        "summary": report.summary,
        "fixed_code": report.fixed_code,
    }


def generate_diff_node(state: BugAnalysisState) -> dict:
    """Node 3: Compares original code and proposed fix to generate a standard Git unified diff."""
    filename = state["filename"]
    print(f"\n[Node 3: generate_diff] Generating non-destructive unified diff for '{filename}'...")

    orig_lines = state["original_code"].splitlines()
    fixed_lines = state["fixed_code"].splitlines()

    diff = "\n".join(
        difflib.unified_diff(
            orig_lines,
            fixed_lines,
            fromfile=f"a/{filename} (original)",
            tofile=f"b/{filename} (proposed patch)",
            lineterm="",
        )
    )

    return {"unified_diff": diff if diff else "(No differences detected.)"}


def human_approval_gate(state: BugAnalysisState) -> dict:
    """Node 4: Human-in-the-Loop (HITL) Review Gate.
    
    Displays the analysis and the proposed patch, then prompts the user for approval.
    """
    print("\n" + "=" * 65)
    print(" 🔍 MINI CODEX CODE REVIEW & BUG REPORT")
    print("=" * 65)
    print(f"Target File : {state['filename']}")
    print(f"Summary     : {state['summary']}\n")

    print("🔴 Confirmed Bugs:")
    if state["confirmed_bugs"]:
        for idx, bug in enumerate(state["confirmed_bugs"], 1):
            print(f"   {idx}. {bug}")
    else:
        print("   (None detected)")

    print("\n🟡 Suspected Issues / Code Smells:")
    if state["suspected_issues"]:
        for idx, issue in enumerate(state["suspected_issues"], 1):
            print(f"   {idx}. {issue}")
    else:
        print("   (None detected)")

    print("\n" + "-" * 65)
    print(" 📝 PROPOSED PATCH (UNIFIED DIFF)")
    print("-" * 65)
    print(state["unified_diff"])
    print("-" * 65)

    # Prompt user for explicit approval (or check CLI flag)
    if "--approve" in sys.argv or "--yes" in sys.argv:
        print("\n👉 Auto-approved via CLI flag (--approve)")
        approved = True
    elif "--reject" in sys.argv or "--no" in sys.argv:
        print("\n👉 Auto-rejected via CLI flag (--reject)")
        approved = False
    else:
        while True:
            try:
                choice = input("\n👉 Do you approve applying this patch? (yes/no): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = "no"

            if choice in ("yes", "y"):
                approved = True
                break
            elif choice in ("no", "n"):
                approved = False
                break
            else:
                print("   ⚠️  Please enter 'yes' to apply the patch, or 'no' to cancel.")

    return {"approved": approved}


def apply_patch_node(state: BugAnalysisState) -> dict:
    """Node 5A: Creates a backup and writes the patch to disk."""
    filename = state["filename"]
    print(f"\n[Node 5A: apply_patch] User APPROVED. Securing backup and writing patch...")

    # 1. Create .bak backup first
    bak_msg = create_backup(filename)
    print(f"  -> {bak_msg}")

    # 2. Apply patch
    patch_msg = apply_patch(filename, state["fixed_code"])
    print(f"  -> {patch_msg}")

    return {"status_message": f"SUCCESS: {patch_msg} (Backup saved as {filename}.bak)"}


def abort_patch_node(state: BugAnalysisState) -> dict:
    """Node 5B: Safely aborts without touching any files."""
    filename = state["filename"]
    print(f"\n[Node 5B: abort_patch] User REJECTED patch. No changes were made.")
    return {"status_message": f"ABORTED: File '{filename}' was left untouched."}


# =====================================================================
# Conditional Edge
# =====================================================================
def route_approval(state: BugAnalysisState) -> Literal["apply_patch", "abort_patch"]:
    """Routes to apply_patch if user approved, or abort_patch if rejected."""
    if state.get("approved", False):
        return "apply_patch"
    return "abort_patch"


# =====================================================================
# Graph Assembly
# =====================================================================
def build_bug_analyzer_graph():
    """Builds and compiles the Bug Analysis LangGraph workflow."""
    workflow = StateGraph(BugAnalysisState)

    # Add nodes
    workflow.add_node("read_target_file", read_target_file)
    workflow.add_node("detect_bugs", detect_bugs_node)
    workflow.add_node("generate_diff", generate_diff_node)
    workflow.add_node("human_approval_gate", human_approval_gate)
    workflow.add_node("apply_patch", apply_patch_node)
    workflow.add_node("abort_patch", abort_patch_node)

    # Add edges
    workflow.add_edge(START, "read_target_file")
    workflow.add_edge("read_target_file", "detect_bugs")
    workflow.add_edge("detect_bugs", "generate_diff")
    workflow.add_edge("generate_diff", "human_approval_gate")

    # Conditional branching based on human approval
    workflow.add_conditional_edges(
        "human_approval_gate",
        route_approval,
        {
            "apply_patch": "apply_patch",
            "abort_patch": "abort_patch",
        },
    )

    workflow.add_edge("apply_patch", END)
    workflow.add_edge("abort_patch", END)

    return workflow.compile()


# Compile graph
bug_graph = build_bug_analyzer_graph()


# =====================================================================
# CLI Runner with Persistent Menu & Error Recovery
# =====================================================================
def select_file(files: list[str]) -> str | None:
    """Persistently prompts the user until a valid file or exit is selected."""
    while True:
        print("\nAvailable files for analysis in sample_project:")
        for idx, f in enumerate(files, 1):
            print(f"  {idx}. {f}")
        print("  (Type 'exit' or 'q' to quit)")

        try:
            selection = input("\nEnter file number or name (default: 1. calculator.py): ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        # Handle defaults and exits
        if not selection:
            return "calculator.py"
        if selection.lower() in ("exit", "quit", "q"):
            return None
        
        # Numeric shortcut (e.g. typing '1' picks 'calculator.py')
        if selection.isdigit() and 1 <= int(selection) <= len(files):
            return files[int(selection) - 1]
        
        # Direct name match (e.g. 'calculator.py' or 'calculator')
        if selection in files:
            return selection
        if f"{selection}.py" in files:
            return f"{selection}.py"

        print(f"\n⚠️  [INVALID SELECTION] '{selection}' is not in sample_project.")
        print(f"   Please enter '1' or '2', or a valid name like 'calculator.py'.")


def main():
    print("=" * 65)
    print("  Mini Codex — Bug Analysis & Patching Workflow (Phase 3)")
    print("=" * 65)

    files = list_python_files()
    if not files:
        print("[ERROR] No Python files found in sample_project!")
        return

    # Check if a filename was provided as a CLI argument for automated runs
    cli_files = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if cli_files and cli_files[0] in files:
        run_file = cli_files[0]
        initial_state = {
            "filename": run_file,
            "original_code": "",
            "confirmed_bugs": [],
            "suspected_issues": [],
            "summary": "",
            "fixed_code": "",
            "unified_diff": "",
            "approved": False,
            "status_message": "",
        }
        result = bug_graph.invoke(initial_state)
        print("\n" + "=" * 65)
        print(f"Workflow Complete: {result['status_message']}")
        print("=" * 65)
        return

    # Interactive Loop: never kills terminal on invalid inputs
    while True:
        target_file = select_file(files)
        if not target_file:
            print("\nGoodbye!")
            break

        initial_state = {
            "filename": target_file,
            "original_code": "",
            "confirmed_bugs": [],
            "suspected_issues": [],
            "summary": "",
            "fixed_code": "",
            "unified_diff": "",
            "approved": False,
            "status_message": "",
        }

        try:
            result = bug_graph.invoke(initial_state)
            print("\n" + "=" * 65)
            print(f"Workflow Complete: {result['status_message']}")
            print("=" * 65)
        except Exception as e:
            print(f"\n❌ [ERROR] An error occurred during analysis: {e}")
            print("   Returning to file selection...")

        # Ask if the user wants to continue or exit
        try:
            again = input("\nWould you like to analyze another file? (yes/no): ").strip().lower()
            if again not in ("yes", "y"):
                print("\nGoodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
