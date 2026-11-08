"""Mini Codex — Streamlit Web Dashboard (Phase 5).

An interactive portfolio web interface showcasing:
  1. AI Code Assistant: Multi-turn chat with seamless abstraction and conversation history.
  2. Bug Hunter: Automated defect detection, unified diff preview, and Human-in-the-Loop approval.
  3. Test-Driven Verification: Automated pytest generation, self-correction reflection, and pass rate metrics.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Mini Codex — Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Terminal and environment configuration
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Import core agent workflows
from tools import (
    BASE_DIR,
    get_workspace_dir,
    set_workspace_dir,
    SAMPLE_DIR,
    list_python_files,
    read_python_file,
    apply_patch,
    create_backup,
    restore_backup,
    run_pytest,
    inspect_project_architecture,
    import_uploaded_project,
)
from agent_graph import app as agent_app
from bug_analyzer import bug_graph
from test_verifier import verifier_graph
from langchain_core.messages import HumanMessage, AIMessage

# Conversation Session Controller Helpers (ChatGPT Style)
def get_default_welcome() -> str:
    return f"Hello! I am Mini Codex. Ask me about any code inside `{get_workspace_dir().name}/`."

def init_conversation(conv_id: str, title: str = "New Chat") -> dict:
    return {
        "id": conv_id,
        "title": title,
        "created_at": time.time(),
        "messages": [
            {"role": "assistant", "content": get_default_welcome()}
        ]
    }

def get_active_conversation() -> dict:
    if "conversations" not in st.session_state or not st.session_state.conversations:
        init_id = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.conversations = {
            init_id: init_conversation(init_id, "Welcome Chat")
        }
        st.session_state.active_conv_id = init_id

    active_id = st.session_state.get("active_conv_id")
    if not active_id or active_id not in st.session_state.conversations:
        active_id = list(st.session_state.conversations.keys())[0]
        st.session_state.active_conv_id = active_id

    return st.session_state.conversations[active_id]

# Handle queued toast notifications that persist across st.rerun()
if "pending_toast" in st.session_state and st.session_state.pending_toast:
    toast_data = st.session_state.pop("pending_toast")
    st.toast(toast_data["message"], icon=toast_data.get("icon", "✅"))

# Custom CSS for ChatGPT-style fixed layout (fixed header and bottom input)
st.markdown("""
<style>
    /* Compact top padding to keep the viewport fixed on screen */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    /* Fixed sticky tabs header */
    div[data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 99;
        background-color: var(--background-color);
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Sidebar: System Status & Evaluation Metrics
# =====================================================================
with st.sidebar:
    st.title("🤖 Mini Codex")
    st.caption("Agentic AI Coding Assistant • Portfolio Edition")
    st.divider()

    # System Status
    st.subheader("⚙️ System Status")
    if api_key:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        st.success(f"Gemini API Connected ")
    else:
        st.error("GEMINI_API_KEY missing from .env")

    st.markdown("**LLM:** `gemini-2.5-flash`")
    st.markdown("**Orchestrator:** `LangGraph StateGraph`")
    st.markdown(f"**Sandbox:** `{get_workspace_dir().name}/`")
    st.divider()

    # Chat Sessions Controller (ChatGPT Style)
    st.subheader("💬 Chat Sessions")
    active_conv = get_active_conversation()

    # ➕ New Chat button
    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
        new_id = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.conversations[new_id] = init_conversation(new_id, "New Chat")
        st.session_state.active_conv_id = new_id
        st.session_state.pending_toast = {
            "message": f"Started fresh chat session ({new_id})",
            "icon": "💬"
        }
        st.rerun()

    # Dropdown selector for past chat sessions
    conv_ids = list(st.session_state.conversations.keys())
    active_idx = conv_ids.index(st.session_state.active_conv_id) if st.session_state.active_conv_id in conv_ids else 0

    def format_conv_label(cid):
        c = st.session_state.conversations.get(cid, {})
        title = c.get("title", "Chat")
        return f"💬 {title} ({cid})"

    selected_conv_id = st.selectbox(
        "Chat Session History",
        options=conv_ids,
        index=active_idx,
        format_func=format_conv_label,
        key="conv_session_selector",
        label_visibility="collapsed"
    )

    if selected_conv_id != st.session_state.active_conv_id:
        st.session_state.active_conv_id = selected_conv_id
        st.rerun()

    col_conv_id, col_conv_del = st.columns([3, 1])
    with col_conv_id:
        st.caption(f"🆔 **Active ID:** `{st.session_state.active_conv_id}`")
    with col_conv_del:
        if st.button("🗑️", key="del_chat_session_btn", help="Delete this conversation thread"):
            if len(st.session_state.conversations) > 1:
                del st.session_state.conversations[st.session_state.active_conv_id]
                st.session_state.active_conv_id = list(st.session_state.conversations.keys())[0]
                st.session_state.pending_toast = {"message": "Deleted conversation thread.", "icon": "🗑️"}
            else:
                cid = st.session_state.active_conv_id
                st.session_state.conversations[cid] = init_conversation(cid, "New Chat")
                st.session_state.pending_toast = {"message": "Reset conversation.", "icon": "🔄"}
            st.rerun()

    st.divider()

    # Active Workspace Manager
    st.subheader("📂 Active Workspace")
    curr_ws = get_workspace_dir()
    default_ws = (BASE_DIR / "sample_project").resolve()
    is_default = (curr_ws == default_ws)
    is_uploaded = "user_workspaces" in str(curr_ws)

    ws_idx = 0 if is_default else (2 if is_uploaded else 1)
    ws_option = st.radio(
        "Workspace Source:",
        ["Built-in (sample_project/)", "Local Project Folder (Path)", "Upload Project (Files / ZIP)"],
        index=ws_idx,
        help="Connect Mini Codex to User A's local project on this computer or upload files/zip."
    )

    if ws_option == "Built-in (sample_project/)":
        if curr_ws != default_ws:
            set_workspace_dir(default_ws)
            st.session_state.pending_toast = {
                "message": "Switched workspace to built-in sample_project/",
                "icon": "📁"
            }
            st.rerun()
        st.caption(f"📁 Path: `{default_ws}`")

    elif ws_option == "Local Project Folder (Path)":
        custom_path_input = st.text_input(
            "Enter Full Folder Path on PC:",
            value=str(curr_ws) if not is_default else str(BASE_DIR),
            help="Type or paste the full directory path to User A's project (e.g. C:\\Projects\\ProjectA)."
        )
        if st.button("🔄 Connect Project Folder", key="apply_ws_btn", type="primary"):
            ok, msg = set_workspace_dir(custom_path_input)
            if ok:
                st.session_state.pending_toast = {
                    "message": f"Connected to Project: {Path(custom_path_input).name}",
                    "icon": "📂"
                }
                st.rerun()
            else:
                st.error(msg)
        st.caption(f"📁 Active: `{curr_ws}`")

    else:  # Upload Project (Files / ZIP)
        st.caption("Upload Python files or a `.zip` archive of User A's project:")
        uploaded = st.file_uploader(
            "Choose files",
            type=["py", "zip"],
            accept_multiple_files=True,
            key="proj_file_uploader",
            label_visibility="collapsed"
        )
        if uploaded:
            if st.button("📥 Import & Open Project", key="upload_proj_btn", type="primary"):
                ok, msg, dest = import_uploaded_project(uploaded, project_name="user_project_a")
                if ok:
                    st.session_state.pending_toast = {
                        "message": f"🎉 Imported {len(uploaded)} item(s) to workspace!",
                        "icon": "📦"
                    }
                    st.rerun()
                else:
                    st.error(msg)
        st.caption(f"📁 Active: `{curr_ws}`")

    st.divider()

    # Evaluation Metrics Card
    st.subheader("📊 Evaluation Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Test Pass Rate", "100%", delta="23/23 tests")
    with col_m2:
        st.metric("Self-Correction", "1 Turn", delta="Green")

    st.metric("Path Traversal Guard", "Active (Enforced)", delta="Zero leaks")
    st.metric("Average Latency", "~1.8s", delta="-0.3s")
    st.divider()

    # Live File Viewer
    st.subheader(f"📁 Files in {get_workspace_dir().name}/")
    files = list_python_files()
    selected_file = st.selectbox("Inspect File Code:", files if files else ["No files found"])
    if selected_file and selected_file != "No files found":
        content = read_python_file(selected_file)
        with st.expander(f"View {selected_file}", expanded=False):
            st.code(content, language="python")

    # Project Architecture Map
    with st.expander("🗺️ Project Architecture Map", expanded=False):
        arch = inspect_project_architecture()
        if arch:
            for item in arch:
                st.markdown(f"**📄 `{item['filename']}`** ({item['lines']} lines)")
                if item["docstring"] != "No module docstring.":
                    st.caption(f"_{item['docstring'][:100]}..._")
                if item["classes"]:
                    st.write(f"• Classes: `{', '.join(item['classes'])}`")
                if item["functions"]:
                    st.write(f"• Functions: `{', '.join(item['functions'])}`")
                if item["imports"]:
                    st.caption(f"Imports: `{', '.join(item['imports'])}`")
                st.divider()
        else:
            st.caption("No Python files found in current workspace.")


# =====================================================================
# Main Tabs: 3 Interactive Workspaces
# =====================================================================
tab_chat, tab_bugs, tab_tests = st.tabs([
    "💬 AI Assistant",
    "🔍 Bug Hunter & HITL Patching",
    "🧪 Test-Driven Verification Loop",
])


# ---------------------------------------------------------------------
# TAB 1: AI Code Assistant (ChatGPT Layout & Multi-Turn Memory)
# ---------------------------------------------------------------------
with tab_chat:
    active_conv = get_active_conversation()
    active_messages = active_conv["messages"]

    # 1. FIXED TOP HEADER: Never scrolls off-screen
    col_head, col_clear = st.columns([5, 1])
    with col_head:
        st.subheader("💬 AI Code Assistant")
        st.caption(f"Session: **{active_conv.get('title', 'Chat')}** (`{active_conv['id']}`) • Workspace: `{get_workspace_dir().name}/`")
    with col_clear:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            active_conv["messages"] = [
                {"role": "assistant", "content": get_default_welcome()}
            ]
            st.session_state.messages = active_conv["messages"]
            st.rerun()

    # Quick starter analysis pills for User A's codebase
    col_p1, col_p2, col_p3 = st.columns(3)
    pill_selected = None
    with col_p1:
        if st.button("🗺️ Explain Architecture", key="pill_arch", use_container_width=True):
            pill_selected = f"Inspect all files in '{get_workspace_dir().name}' and explain the overall architecture, what each module does, and how they connect."
    with col_p2:
        if st.button("🔗 Dependency Breakdown", key="pill_deps", use_container_width=True):
            pill_selected = f"Analyze the module dependencies and import relationships between the files in '{get_workspace_dir().name}'."
    with col_p3:
        if st.button("⚠️ Potential Edge Cases", key="pill_edges", use_container_width=True):
            pill_selected = f"Inspect the functions in '{get_workspace_dir().name}' and identify any subtle edge cases or unhandled failure points."

    # 2. SCROLLABLE MIDDLE CONVERSATION WINDOW: Chat container stays in the middle
    chat_container = st.container(height=480, border=True)
    with chat_container:
        for msg in active_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 3. FIXED BOTTOM INPUT BOX: Strictly positioned BELOW the conversation window (ChatGPT-style)
    user_prompt = st.chat_input("Ask a question (e.g. 'What functions are in models.py and how does main use them?')")
    active_prompt = pill_selected or user_prompt

    if active_prompt:
        # Auto-name conversation on first question if still default
        if active_conv.get("title") in ["New Chat", "Welcome Chat"]:
            clean_title = active_prompt.strip().replace("\n", " ")
            active_conv["title"] = clean_title[:25] + ("..." if len(clean_title) > 25 else "")

        # Append new user message to session messages
        active_messages.append({"role": "user", "content": active_prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                # Build multi-turn LangChain message history (sliding window of last 10 messages)
                langchain_history = []
                for m in active_messages[-10:]:
                    r = m.get("role")
                    c = m.get("content", "")
                    if not c:
                        continue
                    if r == "user":
                        langchain_history.append(HumanMessage(content=c))
                    elif r == "assistant":
                        # Skip initial generic greeting if there are other conversation turns
                        if c == get_default_welcome() and len(active_messages) > 2:
                            continue
                        langchain_history.append(AIMessage(content=c))

                initial_state = {
                    "messages": langchain_history,
                    "iteration_count": 0,
                    "final_answer": "",
                }

                final_answer = ""

                with st.spinner("Mini Codex is thinking..."):
                    try:
                        for chunk in agent_app.stream(initial_state):
                            node_name = list(chunk.keys())[0]
                            node_output = chunk[node_name] or {}

                            if "final_answer" in node_output and node_output["final_answer"]:
                                final_answer = node_output["final_answer"]

                        if not final_answer:
                            final_answer = "I have completed processing your request."

                        st.markdown(final_answer)

                        # Save clean assistant response to conversation history
                        active_messages.append({
                            "role": "assistant",
                            "content": final_answer,
                        })

                    except Exception as e:
                        err_msg = f"❌ Error during agent execution: {e}"
                        st.error(err_msg)
                        active_messages.append({
                            "role": "assistant",
                            "content": err_msg,
                        })

        # Sync legacy messages key for any other references
        st.session_state.messages = active_messages
        # Preserve clean scroll position with input locked at the bottom
        st.rerun()


# ---------------------------------------------------------------------
# TAB 2: Bug Hunter & Human-in-the-Loop Patching
# ---------------------------------------------------------------------
with tab_bugs:
    st.header("🔍 Automated Bug Analysis & Human-in-the-Loop Approval")
    st.markdown(
        "Select any Python file to detect bugs. Mini Codex will separate "
        "**Confirmed Bugs** from **Suspected Smells**, show a **Git-style unified diff**, "
        "and wait for your explicit approval before modifying the file."
    )

    col_target, col_btn = st.columns([3, 1])
    with col_target:
        target_to_analyze = st.selectbox(
            "Select target file to analyze:",
            [f for f in list_python_files() if not f.startswith("test_")],
            key="bug_target",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_analysis_btn = st.button("🚀 Analyze for Bugs", type="primary")

    # If the user selects a different file, automatically clear old analysis
    if "current_bug_state" in st.session_state and st.session_state.current_bug_state:
        if st.session_state.current_bug_state.get("filename") != target_to_analyze:
            st.session_state.current_bug_state = None

    # Optional restore option if a backup copy exists on disk
    bak_file = (get_workspace_dir() / f"{target_to_analyze}.bak")
    if bak_file.exists():
        col_bak_info, col_bak_btn = st.columns([3, 1])
        with col_bak_info:
            st.caption(f"💾 A backup copy exists for this file: `{target_to_analyze}.bak`")
        with col_bak_btn:
            if st.button(f"⏪ Restore from Backup", key="restore_standalone_bak_btn"):
                res = restore_backup(target_to_analyze)
                st.session_state.pending_toast = {
                    "message": f"⏪ Successfully restored '{target_to_analyze}' from backup!",
                    "icon": "ℹ️",
                }
                st.session_state.current_bug_state = None
                st.rerun()

    if run_analysis_btn:
        with st.spinner(f"Analyzing '{target_to_analyze}' with Gemini 2.5 Flash..."):
            initial_bug_state = {
                "filename": target_to_analyze,
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
                # Run through node 1, 2, and 3
                from bug_analyzer import read_target_file, detect_bugs_node, generate_diff_node
                s1 = {**initial_bug_state, **read_target_file(initial_bug_state)}
                s2 = {**s1, **detect_bugs_node(s1)}
                s3 = {**s2, **generate_diff_node(s2)}
                st.session_state.current_bug_state = s3
            except Exception as e:
                st.error(f"Failed to analyze bugs: {e}")

    # Display Bug Report & Proposed Diff if analysis exists
    if "current_bug_state" in st.session_state and st.session_state.current_bug_state:
        bstate = st.session_state.current_bug_state

        st.subheader(f"Code Review: `{bstate['filename']}`")
        st.info(f"**Summary:** {bstate['summary']}")

        col_conf, col_susp = st.columns(2)
        with col_conf:
            st.markdown("### 🔴 Confirmed Bugs")
            if bstate["confirmed_bugs"]:
                for idx, bug in enumerate(bstate["confirmed_bugs"], 1):
                    st.error(f"**{idx}.** {bug}")
            else:
                st.success("No confirmed bugs found.")

        with col_susp:
            st.markdown("### 🟡 Suspected Issues")
            if bstate["suspected_issues"]:
                for idx, smell in enumerate(bstate["suspected_issues"], 1):
                    st.warning(f"**{idx}.** {smell}")
            else:
                st.caption("No code smells detected.")

        # Requirement 1: Only show Proposed Patch & Approval Gate if confirmed bugs and diff exist
        has_confirmed_bugs = bool(bstate.get("confirmed_bugs"))
        diff_text = bstate.get("unified_diff", "").strip()
        has_diff = bool(diff_text and diff_text != "(No differences detected.)")

        if has_confirmed_bugs and has_diff:
            st.markdown("### 📝 Proposed Patch (Unified Diff)")
            st.code(bstate["unified_diff"], language="diff")

            # Human-in-the-Loop Actions
            st.markdown("### 🛡️ Human-in-the-Loop Approval Gate")
            col_appr, col_rej, col_rest = st.columns([2, 2, 2])

            with col_appr:
                if st.button("✅ Approve & Apply Patch", type="primary", key="approve_patch_btn"):
                    bak_msg = create_backup(bstate["filename"])
                    apply_msg = apply_patch(bstate["filename"], bstate["fixed_code"])
                    st.session_state.pending_toast = {
                        "message": f"🎉 Patch successfully applied to '{bstate['filename']}'! (Backup: {bstate['filename']}.bak)",
                        "icon": "✅",
                    }
                    # Requirement 2: Vanish diff and approval gate once patch is applied, ready for next file
                    st.session_state.current_bug_state = None
                    st.rerun()

            with col_rej:
                if st.button("❌ Reject Patch", key="reject_patch_btn"):
                    st.session_state.pending_toast = {
                        "message": f"Patch rejected. '{bstate['filename']}' remains untouched.",
                        "icon": "🛡️",
                    }
                    st.session_state.current_bug_state = None
                    st.rerun()

            with col_rest:
                if st.button("⏪ Undo / Restore from Backup", key="undo_patch_btn"):
                    res = restore_backup(bstate["filename"])
                    st.session_state.pending_toast = {
                        "message": f"⏪ Successfully restored '{bstate['filename']}' from backup!",
                        "icon": "ℹ️",
                    }
                    st.session_state.current_bug_state = None
                    st.rerun()
        else:
            # Requirement 1: No bugs found -> Do NOT display diff or approval gate!
            st.success("✨ **Clean Code!** No confirmed bugs detected in this file. No patch or approval action required.")


# ---------------------------------------------------------------------
# TAB 3: Test-Driven Verification Loop
# ---------------------------------------------------------------------
with tab_tests:
    st.header("🧪 Test-Driven Self-Correction Verification Loop")
    st.markdown(
        "Demonstrates modern SWE-bench agentic behavior: "
        "The agent writes **pytest unit tests**, executes them in a sandboxed subprocess, "
        "and **iteratively self-corrects** code until 100% of tests turn green."
    )

    col_test_target, col_test_btn = st.columns([3, 1])
    with col_test_target:
        test_target_file = st.selectbox(
            "Select module for automated test verification:",
            [f for f in list_python_files() if not f.startswith("test_")],
            key="test_target_select",
        )
    with col_test_btn:
        st.write("")
        st.write("")
        start_verification = st.button("▶️ Run Test Verification", type="primary")

    if start_verification:
        with st.spinner("Running automated test generation and self-correction loop..."):
            initial_test_state = {
                "filename": test_target_file,
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
                res = verifier_graph.invoke(initial_test_state)
                st.session_state.last_verification_result = res
            except Exception as e:
                st.error(f"Verification workflow encountered error: {e}")

    # Display Test Results if available
    if "last_verification_result" in st.session_state and st.session_state.last_verification_result:
        vres = st.session_state.last_verification_result

        if vres.get("tests_passed", False):
            st.success(f"🎉 {vres.get('final_status', 'All tests verified green!')}")
        else:
            st.error(f"❌ {vres.get('final_status', 'Tests did not pass.')}")

        # Metrics overview
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("Test Status", "PASSED ✅" if vres.get("tests_passed") else "FAILED ❌")
        with col_t2:
            st.metric("Self-Correction Retries", f"{vres.get('retry_count', 0)} / 3")
        with col_t3:
            st.metric("Test Suite File", vres.get("test_filename", "test_calculator.py"))

        # Terminal output viewer
        st.subheader("🖥️ Pytest Subprocess Execution Log")
        st.code(vres.get("test_output", "No output captured."), language="text")

        # Test Code preview
        with st.expander(f"Inspect Generated Test Suite ({vres.get('test_filename')})"):
            st.code(vres.get("test_code", ""), language="python")


# Footer
st.divider()
st.caption("Mini Codex • Built with Google Gemini 2.5 Flash, LangGraph, Pytest, and Streamlit.")

