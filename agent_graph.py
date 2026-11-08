"""Mini Codex — Phase 2 LangGraph Application.

Refactors the agent into a state-driven workflow graph with:
  START -> analyze_request -> [conditional edge: should_continue]
            ├─> execute_tool -> process_tool_result -> analyze_request (loop)
            └─> final_response -> END
"""

import os
import sys
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import our safe tools from Phase 1
from tools import list_python_files, read_python_file

# 1. Terminal configuration for Windows PowerShell (UTF-8 encoding)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] GEMINI_API_KEY is missing from .env file.")
    sys.exit(1)

# 3. Maximum iteration limit to prevent infinite loops
MAX_ITERATIONS = 5

# 4. Wrap local tools into LangChain tools
langchain_tools = [tool(list_python_files), tool(read_python_file)]
TOOL_MAP = {t.name: t for t in langchain_tools}

# 5. Initialize the Model with tools bound
SYSTEM_PROMPT = """You are Mini Codex, a precise AI coding assistant built on LangGraph.
You help users inspect, explain, and debug Python files strictly in the active workspace project.
Always use your tools (list_python_files, read_python_file) to inspect ground-truth code before answering.
Never guess what code is in a file without reading it first."""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    api_key=api_key,
)
llm_with_tools = llm.bind_tools(langchain_tools)


# =====================================================================
# State Definition
# =====================================================================
class AgentState(TypedDict):
    """The shared memory state passed through all nodes in the graph."""

    # add_messages is a LangGraph reducer: it appends new messages rather than overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    # Tracks how many tool cycles have been performed to prevent infinite loops
    iteration_count: int
    # Stores the clean final string answer presented to the user
    final_answer: str


# =====================================================================
# Graph Nodes
# =====================================================================
def analyze_request(state: AgentState) -> dict:
    """Node 1: Analyze Request.
    
    Calls Gemini with the current message history and bound tools.
    Returns the model's AIMessage (which may contain text, tool calls, or both).
    """
    print("\n[Node 1: analyze_request] Querying Gemini with current state...")
    messages = state["messages"]

    # Prepend dynamic system instruction with workspace context and conversational memory guidance
    if not isinstance(messages[0], SystemMessage):
        try:
            from tools import get_workspace_dir, list_python_files
            ws_name = get_workspace_dir().name
            available_files = list_python_files()
            ws_context = (
                f"\n\nActive Project Workspace: '{ws_name}'\n"
                f"Available Files in Workspace: {available_files}\n"
                "Multi-Turn Context & Grounding:\n"
                "- Maintain full conversational context across previous user & assistant turns.\n"
                "- When the user says 'it', 'this file', 'explain it', or asks follow-up questions, resolve references using earlier conversation turns.\n"
                "- Always inspect real code using read_python_file before answering questions about files."
            )
            dynamic_sys_prompt = SYSTEM_PROMPT + ws_context
        except Exception:
            dynamic_sys_prompt = SYSTEM_PROMPT

        messages = [SystemMessage(content=dynamic_sys_prompt)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def execute_tool(state: AgentState) -> dict:
    """Node 2: Select and Execute Tool.
    
    Inspects the tool_calls on the latest AIMessage, executes the requested
    Python function locally, and wraps the outputs as ToolMessages.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = []

    print(f"\n[Node 2: execute_tool] Executing {len(tool_calls)} requested tool call(s)...")

    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        call_id = call["id"]

        print(f"  -> Dispatched Tool : {name}")
        print(f"  -> Arguments       : {args}")

        if name in TOOL_MAP:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as e:
                result = f"Tool Execution Error: {str(e)}"
        else:
            result = f"Error: Tool '{name}' not found."

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call_id, name=name)
        )

    current_count = state.get("iteration_count", 0) + 1
    return {
        "messages": tool_messages,
        "iteration_count": current_count,
    }


def process_tool_result(state: AgentState) -> dict:
    """Node 3: Process Tool Result.
    
    Inspects and logs tool outputs for observability and debugging before
    passing control back to analyze_request.
    """
    last_message = state["messages"][-1]
    preview = str(last_message.content)
    if len(preview) > 180:
        preview = preview[:180] + "... [truncated]"

    print(f"\n[Node 3: process_tool_result] Captured tool output (Turn {state.get('iteration_count', 0)}):")
    print(f"  -> Output Preview  : {preview}")

    # Return state metadata to ensure stream chunk returns a valid dictionary
    return {"iteration_count": state.get("iteration_count", 0)}


def final_response(state: AgentState) -> dict:
    """Node 4: Final Response.
    
    Synthesizes the terminal output when the model is done or iteration limit is reached.
    """
    print("\n[Node 4: final_response] Extracting final assistant answer...")
    last_message = state["messages"][-1]

    # If the last message is an AIMessage, extract its text content
    raw_content = getattr(last_message, "content", "")
    if isinstance(raw_content, list):
        # Handle list of content blocks (e.g. [{'type': 'text', 'text': '...'}])
        text_parts = []
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            else:
                text_parts.append(str(block))
        answer = "\n".join(text_parts)
    else:
        answer = str(raw_content)

    # Check if we terminated because of loop limits
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        answer += "\n\n[Note: Terminated early because maximum tool iterations was reached.]"

    return {"final_answer": answer}


# =====================================================================
# Conditional Edge (Router)
# =====================================================================
def should_continue(state: AgentState) -> Literal["execute_tool", "final_response"]:
    """Determines whether the graph should branch to a tool execution or finalize."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    current_iterations = state.get("iteration_count", 0)

    # If model requested a tool and we haven't reached the safety limit -> run tool
    if tool_calls and current_iterations < MAX_ITERATIONS:
        return "execute_tool"

    # Otherwise -> finalize response
    return "final_response"


# =====================================================================
# Graph Construction & Compilation
# =====================================================================
def build_graph():
    """Constructs and compiles the StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add all functional nodes
    workflow.add_node("analyze_request", analyze_request)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("process_tool_result", process_tool_result)
    workflow.add_node("final_response", final_response)

    # 2. Add fixed and conditional edges
    # START -> analyze_request
    workflow.add_edge(START, "analyze_request")

    # analyze_request -> (execute_tool OR final_response)
    workflow.add_conditional_edges(
        "analyze_request",
        should_continue,
        {
            "execute_tool": "execute_tool",
            "final_response": "final_response",
        },
    )

    # execute_tool -> process_tool_result -> analyze_request (Feedback Loop)
    workflow.add_edge("execute_tool", "process_tool_result")
    workflow.add_edge("process_tool_result", "analyze_request")

    # final_response -> END
    workflow.add_edge("final_response", END)

    # Compile the graph
    return workflow.compile()


# Compile the graph application once at startup
app = build_graph()


# =====================================================================
# Interactive Terminal Runner
# =====================================================================
def run_query(question: str):
    """Executes a single user question through the compiled LangGraph."""
    print("=" * 60)
    print(f"LangGraph Workflow Started: '{question}'")
    print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration_count": 0,
        "final_answer": "",
    }

    # Execute the graph
    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Mini Codex (LangGraph) Final Answer:")
    print("=" * 60)
    print(result.get("final_answer", "").strip())
    print("=" * 60)


def main():
    print("=" * 60)
    print("  Mini Codex — LangGraph AI Assistant (Phase 2)")
    print("=" * 60)
    print("Ask questions about sample_project code.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Mini-Codex (Graph) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        run_query(user_input)


if __name__ == "__main__":
    main()
