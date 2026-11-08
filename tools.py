"""Tools module for Mini Codex.

Provides safe, sandboxed filesystem tools for inspecting, backing up,
patching, and running pytest tests strictly within the designated sample_project directory.
"""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# Define the strictly permitted directory (absolute, resolved path)
BASE_DIR = Path(__file__).parent.resolve()
CURRENT_WORKSPACE = (BASE_DIR / "sample_project").resolve()
SAMPLE_DIR = CURRENT_WORKSPACE  # Backwards compatibility alias


def get_workspace_dir() -> Path:
    """Return the currently active workspace directory path."""
    return CURRENT_WORKSPACE


def set_workspace_dir(new_path: str | Path) -> tuple[bool, str]:
    """Change the active workspace directory for Mini Codex.
    
    Args:
        new_path: Directory path to set as active workspace.
        
    Returns:
        tuple[bool, str]: (Success boolean, Status message)
    """
    global CURRENT_WORKSPACE, SAMPLE_DIR
    p = Path(new_path).resolve()
    if not p.exists():
        return False, f"Directory does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"
    CURRENT_WORKSPACE = p
    SAMPLE_DIR = p
    return True, f"Workspace set to: {p}"


def inspect_project_architecture() -> list[dict]:
    """Parse AST of all Python files in the active workspace to extract structure."""
    workspace = get_workspace_dir()
    files = list_python_files(recursive=True)
    results = []

    for rel_path in files:
        target_path = (workspace / rel_path).resolve()
        if not target_path.exists() or not target_path.is_file():
            continue
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            
            docstring = ast.get_docstring(tree) or "No module docstring."
            classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    imports.append(mod)

            results.append({
                "filename": rel_path,
                "docstring": docstring,
                "classes": classes,
                "functions": functions,
                "imports": sorted(set(imports)),
                "lines": len(content.splitlines()),
            })
        except Exception:
            results.append({
                "filename": rel_path,
                "docstring": "Could not parse AST.",
                "classes": [],
                "functions": [],
                "imports": [],
                "lines": 0,
            })

    return results


def import_uploaded_project(uploaded_files, project_name: str = "imported_project") -> tuple[bool, str, Path]:
    """Save uploaded files or extract a zip archive into user_workspaces/project_name.
    
    Args:
        uploaded_files: Single uploaded file or list of uploaded files from st.file_uploader.
        project_name: Subfolder name to save the project in.
        
    Returns:
        tuple[bool, str, Path]: (Success, message, destination_path)
    """
    dest_dir = (BASE_DIR / "user_workspaces" / project_name).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    saved_count = 0
    for uploaded in uploaded_files:
        if not uploaded:
            continue
        file_name = uploaded.name
        if file_name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(uploaded, "r") as zf:
                    zf.extractall(dest_dir)
                saved_count += 1
            except Exception as e:
                return False, f"Failed to extract zip file: {e}", dest_dir
        elif file_name.lower().endswith(".py"):
            target_path = (dest_dir / file_name).resolve()
            try:
                with open(target_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                saved_count += 1
            except Exception as e:
                return False, f"Failed to save file '{file_name}': {e}", dest_dir

    if saved_count == 0:
        return False, "No valid Python files or ZIP archives found in upload.", dest_dir

    set_workspace_dir(dest_dir)
    return True, f"Successfully imported {saved_count} item(s) to '{dest_dir.name}'.", dest_dir


def _validate_safe_path(filename: str) -> tuple[Path | None, str | None]:
    """Internal helper to enforce extension and path traversal safety.
    
    Returns:
        tuple[Path | None, str | None]: (target_path, None) if safe,
                                        (None, error_message) if access denied.
    """
    if not filename.lower().endswith(".py"):
        return None, f"Access Denied: Only Python source files (.py) are permitted. Received: '{filename}'"

    clean_name = filename.strip().lstrip("/\\")
    workspace = get_workspace_dir()
    target_path = (workspace / clean_name).resolve()

    try:
        target_path.relative_to(workspace)
    except ValueError:
        return None, f"Access Denied: Path traversal detected. '{filename}' is outside permitted directory."

    return target_path, None


IGNORED_PATTERNS = {
    ".git", ".venv", "venv", "myenv", "env", "envs", "virtualenv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", "node_modules", "dist", "build", ".egg-info",
    "site-packages", "dist-packages", "lib", "libs", "include", "scripts", "bin"
}


def _is_ignored_path(file_path: Path) -> bool:
    """Check if any folder in the path is a virtual environment, library, or build cache."""
    for part in file_path.parts:
        part_lower = part.lower()
        if part_lower.startswith("."):
            return True
        if part_lower in IGNORED_PATTERNS:
            return True
        if "site-packages" in part_lower or "dist-packages" in part_lower:
            return True
        if part_lower.endswith("env") or part_lower.endswith("venv"):
            return True
    return False


def list_python_files(recursive: bool = True) -> list[str]:
    """List all Python (.py) source files available in the active workspace directory.
    
    Args:
        recursive: Whether to scan subdirectories while ignoring build/env directories.
        
    Returns:
        list[str]: Relative path strings of permitted Python files.
    """
    workspace = get_workspace_dir()
    if not workspace.exists() or not workspace.is_dir():
        return []
    
    py_files = []
    if recursive:
        for file_path in workspace.rglob("*.py"):
            # Exclude files in virtualenvs, third-party libraries, caches, or backup files
            if _is_ignored_path(file_path):
                continue
            if file_path.name.endswith(".bak"):
                continue
            try:
                rel_path = file_path.relative_to(workspace)
                py_files.append(str(rel_path).replace("\\", "/"))
            except ValueError:
                continue
    else:
        for file in workspace.iterdir():
            if file.is_file() and file.suffix.lower() == ".py" and not file.name.endswith(".bak"):
                py_files.append(file.name)
                
    return sorted(py_files)


def read_python_file(filename: str) -> str:
    """Read and return the source code of a specified Python (.py) file in the active workspace.
    
    Args:
        filename: Relative path of the Python file to read (e.g., 'calculator.py' or 'utils/helpers.py').

    Returns:
        str: The raw content of the file, or an error message if access is denied.
    """
    target_path, error = _validate_safe_path(filename)
    if error:
        return error

    if not target_path.exists():
        return f"Error: File '{filename}' does not exist in workspace."

    if not target_path.is_file():
        return f"Error: '{filename}' is not a regular file."

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


def create_backup(filename: str) -> str:
    """Create a backup copy of a Python file with a .bak extension before modifying it.
    
    Args:
        filename: The name of the Python file to back up (e.g., 'calculator.py').

    Returns:
        str: Success message with backup filename, or an error message.
    """
    target_path, error = _validate_safe_path(filename)
    if error:
        return error

    if not target_path.exists() or not target_path.is_file():
        return f"Error: Cannot back up non-existent file '{filename}'."

    backup_path = target_path.with_name(f"{target_path.name}.bak")
    try:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, backup_path)
        return f"Backup created successfully: '{backup_path.name}'"
    except Exception as e:
        return f"Error creating backup for '{filename}': {str(e)}"


def apply_patch(filename: str, new_code: str) -> str:
    """Safely write new source code into a Python file inside the active workspace.
    
    Args:
        filename: The relative path of the Python file to overwrite.
        new_code: The complete updated Python source code.

    Returns:
        str: Success message or error details.
    """
    target_path, error = _validate_safe_path(filename)
    if error:
        return error

    if not target_path.exists() or not target_path.is_file():
        return f"Error: Target file '{filename}' does not exist."

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        return f"Patch successfully applied to '{filename}'."
    except Exception as e:
        return f"Error applying patch to '{filename}': {str(e)}"


def restore_backup(filename: str) -> str:
    """Restore a Python file from its .bak backup copy if available.
    
    Args:
        filename: The name of the Python file to restore (e.g., 'calculator.py').

    Returns:
        str: Success message or error details.
    """
    target_path, error = _validate_safe_path(filename)
    if error:
        return error

    backup_path = target_path.with_name(f"{target_path.name}.bak")
    if not backup_path.exists() or not backup_path.is_file():
        return f"Error: No backup found for '{filename}'."

    try:
        shutil.copy2(backup_path, target_path)
        return f"Successfully restored '{filename}' from backup '{backup_path.name}'."
    except Exception as e:
        return f"Error restoring backup: {str(e)}"


def delete_backup(filename: str) -> bool:
    """Safely delete the .bak backup copy of a Python file if it exists.
    
    Args:
        filename: Relative name or path of the Python file.
        
    Returns:
        bool: True if deleted or not present, False on error.
    """
    target_path, error = _validate_safe_path(filename)
    if error or not target_path:
        return False

    backup_path = target_path.with_name(f"{target_path.name}.bak")
    if backup_path.exists():
        try:
            backup_path.unlink(missing_ok=True)
            return True
        except Exception:
            return False
    return True


def write_test_file(filename: str, test_code: str) -> str:
    """Safely write a pytest test file into the active workspace directory.
    
    Args:
        filename: Name of the test file (base filename must start with 'test_' and end with '.py').
        test_code: The Python test suite code to write.
        
    Returns:
        str: Success message or error details.
    """
    clean_filename = filename.replace("\\", "/")
    base_name = Path(clean_filename).name
    if not base_name.startswith("test_") or not base_name.endswith(".py"):
        return f"Access Denied: Test files must follow 'test_*.py' naming convention. Received: '{filename}'"

    target_path, error = _validate_safe_path(clean_filename)
    if error:
        return error

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        return f"Test file successfully written to '{clean_filename}'."
    except Exception as e:
        return f"Error writing test file '{clean_filename}': {str(e)}"


def run_pytest(test_filename: str) -> dict:
    """Safely execute pytest against a specific test file strictly in the active workspace.
    
    Args:
        test_filename: Relative path of the test file to run (e.g., 'test_calculator.py').
        
    Returns:
        dict: {
            'passed': bool,
            'exit_code': int,
            'output': str,
            'error': str | None
        }
    """
    clean_filename = test_filename.replace("\\", "/")
    base_name = Path(clean_filename).name
    if not base_name.startswith("test_") or not base_name.endswith(".py"):
        return {
            "passed": False,
            "exit_code": -1,
            "output": "",
            "error": f"Access Denied: Only 'test_*.py' files can be executed. Received: '{test_filename}'",
        }

    target_path, error = _validate_safe_path(clean_filename)
    if error:
        return {"passed": False, "exit_code": -1, "output": "", "error": error}

    if not target_path.exists():
        return {
            "passed": False,
            "exit_code": -1,
            "output": "",
            "error": f"File '{test_filename}' does not exist.",
        }

    # Execute pytest in a controlled subprocess pointing strictly at the target file
    cmd = [sys.executable, "-m", "pytest", str(target_path), "-v", "--no-header"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(get_workspace_dir()),
            capture_output=True,
            text=True,
            timeout=20,  # 20 second timeout protection
        )
        passed = (proc.returncode == 0)
        output = proc.stdout + proc.stderr
        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "output": output.strip(),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "exit_code": -1,
            "output": "",
            "error": "Test execution timed out after 20 seconds.",
        }
    except Exception as e:
        return {
            "passed": False,
            "exit_code": -1,
            "output": "",
            "error": f"Subprocess error executing pytest: {str(e)}",
        }


# =====================================================================
# Manual Test Harness (Verification Step)
# =====================================================================
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("--- Running Tools Self-Test (Phase 4) ---")
    
    # 1. Test listing files
    files = list_python_files()
    print(f"\n1. list_python_files(): {files}")
    assert "calculator.py" in files

    # 2. Test writing a temporary test file
    print("\n2. write_test_file('test_dummy.py'):")
    sample_test = """
def test_dummy_add():
    assert 2 + 2 == 4
"""
    write_res = write_test_file("test_dummy.py", sample_test)
    print(f"   Result: {write_res}")
    assert "successfully written" in write_res

    # 3. Test running pytest against the dummy test
    print("\n3. run_pytest('test_dummy.py'):")
    pytest_res = run_pytest("test_dummy.py")
    print(f"   Passed: {pytest_res['passed']}")
    print(f"   Exit Code: {pytest_res['exit_code']}")
    print(f"   Output Preview:\n{pytest_res['output'][:160]}...")
    assert pytest_res["passed"] is True
    assert pytest_res["exit_code"] == 0

    # 4. Clean up dummy test file
    dummy_path = get_workspace_dir() / "test_dummy.py"
    if dummy_path.exists():
        dummy_path.unlink()

    # 5. Security: test running non-test file (should be blocked)
    print("\n4. Security Test: run_pytest('calculator.py'):")
    blocked = run_pytest("calculator.py")
    print(f"   Blocked: {blocked['error']}")
    assert "Access Denied" in blocked["error"]

    print("\n[SUCCESS] All Phase 4 test runner tools verified successfully!")
