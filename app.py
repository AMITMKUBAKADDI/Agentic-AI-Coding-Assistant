"""Mini Codex — Phase 1 Application.

A beginner-friendly agentic coding assistant that can inspect, read,
and explain Python files in the sample_project folder using tool-calling.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import list_python_files, read_python_file

# 1. Terminal configuration for Windows PowerShell (UTF-8)
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

# 3. Initialize Gemini Client
client = genai.Client(api_key=api_key)

# 4. Define Tool Registry
# We give Gemini the function signatures and map their names to Python callables.
TOOLS = [list_python_files, read_python_file]
TOOL_MAP = {
    "list_python_files": list_python_files,
    "read_python_file": read_python_file,
}

SYSTEM_INSTRUCTION = """You are Mini Codex, a helpful and precise AI coding assistant.
Your goal is to help users understand, analyze, and inspect Python code in their project.
You have access to safe tools to list and read Python files in the sample_project directory.
Always inspect the relevant code using your tools before answering questions about it.
Do not guess the contents of files—always read them first.
"""

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=TOOLS,
    # We disable automatic function calling so WE can inspect each step manually!
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    temperature=0.1,
)


def run_agent(user_question: str) -> None:
    """Orchestrate the agent loop for a single user question.
    
    1. Sends question to Gemini.
    2. If Gemini requests a tool call, displays it and executes it in Python.
    3. Sends the tool result back to Gemini.
    4. Repeats until Gemini produces the final answer (or hits max iterations).
    """
    print(f"\n[User Question]: {user_question}")
    
    # Initialize conversation history with user's question
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)]
        )
    ]
    
    max_turns = 5
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # Call Gemini
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=CONFIG,
            )
        except Exception as e:
            print(f"\n[ERROR] Gemini API request failed: {e}")
            return

        # Check if Gemini decided to call any tools
        if not response.function_calls:
            # No tool calls requested: Gemini has formulated its final answer!
            print("\n" + "=" * 50)
            print("[Mini Codex Final Answer]:")
            print("=" * 50)
            print(response.text.strip())
            print("=" * 50)
            return

        # Append Gemini's response (containing function_call) to conversation history
        contents.append(response.candidates[0].content)

        # Process each tool call requested by the model
        for call in response.function_calls:
            tool_name = call.name
            tool_args = call.args or {}

            # Display the selected tool and its arguments (Requirement 9)
            print(f"\n[Agent Decision - Turn {turn}]: Model requested tool execution.")
            print(f"  -> Tool Selected : {tool_name}")
            print(f"  -> Arguments     : {tool_args}")

            # Execute the tool in Python
            if tool_name in TOOL_MAP:
                func = TOOL_MAP[tool_name]
                try:
                    tool_output = func(**tool_args)
                except Exception as ex:
                    tool_output = f"Execution Error in {tool_name}: {str(ex)}"
            else:
                tool_output = f"Error: Tool '{tool_name}' is not recognized."

            # Display the tool execution result (Requirement 9)
            result_preview = str(tool_output)
            if len(result_preview) > 200:
                result_preview = result_preview[:200] + "... [truncated]"
            print(f"  -> Tool Result   : {result_preview}")

            # Send the tool result back to Gemini
            tool_response_part = types.Part.from_function_response(
                name=tool_name,
                response={"result": tool_output}
            )
            contents.append(
                types.Content(
                    role="tool",
                    parts=[tool_response_part]
                )
            )

    print("\n[WARNING] Reached maximum allowed turns without a final response.")


def main():
    """Terminal interactive interface."""
    print("=" * 60)
    print("  Mini Codex — AI Coding Assistant (Phase 1)")
    print("=" * 60)
    print("Ask questions about code in sample_project (e.g.):")
    print("  • 'What files are in this project?'")
    print("  • 'What functions are defined in calculator.py?'")
    print("  • 'Explain how main.py works.'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Mini-Codex > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        run_agent(user_input)


if __name__ == "__main__":
    main()

