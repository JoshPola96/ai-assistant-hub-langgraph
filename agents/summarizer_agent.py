# summarizer_agent.py

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.summarizer_tools import summarize_text
from utils.file_service import read_uploaded_file

load_dotenv()

# --- Agent State ---
class SummarizerState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], add_messages]
    reasoning_log: List[str]
    file_path: str
    chat_history: str

# --- Tool: File Reader ---
def call_file_reader(state: SummarizerState):
    file_path = state.get("file_path", "").strip()
    if not file_path:
        raise ValueError("No file path provided.")
    content = read_uploaded_file(file_path)
    state["reasoning_log"].append(f"📄 Read file: `{file_path}` ({len(content)} chars)")
    return {
        "messages": [ToolMessage(tool_call_id="file_reader", content=content)],
        "reasoning_log": state["reasoning_log"],
        "file_path": file_path,
        "chat_history": state.get("chat_history", "")
    }

# --- Tool: Summarize File or Chat ---
def call_summarizer(state: SummarizerState):
    content = ""
    if any(isinstance(m, ToolMessage) for m in state["messages"]):
        content = next(m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)).content
    elif state.get("chat_history"):
        content = state["chat_history"]
        state["reasoning_log"].append("💬 Using chat history for summarization.")

    if len(content) > 10000:
        content = content[:10000]
        state["reasoning_log"].append(f"✂️ Original length: {len(content)}. Truncated to 10,000 characters.")

    summary = summarize_text.invoke({"text": content, "length": "medium"})
    state["reasoning_log"].append("🧠 Generated summary using `summarize_text`.")
    return {
        "messages": [AIMessage(content=summary)],
        "reasoning_log": state["reasoning_log"],
        "file_path": state.get("file_path", ""),
        "chat_history": state.get("chat_history", "")
    }

# --- New: Summarize Chat History directly ---
def summarize_chat_history(state: SummarizerState):
    return call_summarizer(state)

# --- Router ---
def route_summary_type(state: SummarizerState) -> str:
    if state.get("file_path"):
        return "read_file"
    elif state.get("chat_history"):
        return "summarize_chat_history"
    else:
        state["reasoning_log"].append("⚠️ Missing file and chat history; returning clarification message.")
        state["messages"].append(AIMessage(content="Please provide a file or chat history to summarize."))
        return END

def router_node(state: SummarizerState) -> SummarizerState:
    state["reasoning_log"].append("🗺️ Routing input to appropriate summarization flow.")
    return state

# --- LangGraph Build ---
workflow = StateGraph(SummarizerState)
workflow.add_node("read_file", call_file_reader)
workflow.add_node("summarize_content", call_summarizer)
workflow.add_node("summarize_chat_history", summarize_chat_history)

workflow.set_entry_point("router")
workflow.add_node("router", router_node)
workflow.add_conditional_edges("router", route_summary_type, {
    "read_file": "read_file",
    "summarize_chat_history": "summarize_chat_history"
})

workflow.add_edge("read_file", "summarize_content")
workflow.add_edge("summarize_chat_history", "summarize_content")
workflow.add_edge("summarize_content", END)

summarizer_agent_app = workflow.compile()

# --- Unified Entrypoint ---
def run_summarizer_agent(
    file_path: str = "",
    chat_history: str = ""
) -> tuple[str, str]:
    initial_state: SummarizerState = {
        "messages": [HumanMessage(content="Summarize input")],
        "reasoning_log": ["🧠 Summarizer Agent Started"],
        "file_path": file_path or "",
        "chat_history": chat_history or ""
    }
    final_state = summarizer_agent_app.invoke(initial_state)
    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    summary_msg = ai_messages[-1] if ai_messages else None
    summary = summary_msg.content if summary_msg else "⚠️ No summary generated."
    log = "\n".join(final_state.get("reasoning_log", []))
    return summary, log

if __name__ == "__main__":
    print("\n====================")
    print("🧪 SUMMARIZER AGENT TEST SUITE")
    print("====================")

    test_cases = [
        {
            "desc": "Summarize short chat history",
            "chat_history": "User: What is machine learning?\nAI: It's a subset of AI...",
            "file_path": ""
        },
        {
            "desc": "Summarize long chat history (should truncate)",
            "chat_history": "Blah. " * 3000,  # ~15K chars
            "file_path": ""
        },
        {
            "desc": "Missing input test",
            "chat_history": "",
            "file_path": ""
        },
        {
            "desc": "File path only test",
            "chat_history": "",
            "file_path": "mock_file.txt"  # simulate existing file
        },
        {
            "desc": "File and chat provided (file prioritized)",
            "chat_history": "Some chat text that should be ignored.",
            "file_path": "mock_file.txt"
        },
        {
            "desc": "Very large file test (simulate)",
            "chat_history": "",
            "file_path": "large_mock_file.txt"  # simulate large content
        },
        {
            "desc": "Empty file test (simulate empty file)",
            "chat_history": "",
            "file_path": "empty_mock_file.txt"
        },
        {
            "desc": "Garbage chat input",
            "chat_history": "@@@!!!###$$$%%%^^^&&&***(((???",
            "file_path": ""
        },
    ]

    for case in test_cases:
        print("\n--------------------")
        print(f"📝 {case['desc']}")
        try:
            summary, log = run_summarizer_agent(
                file_path=case.get("file_path", ""),
                chat_history=case.get("chat_history", "")
            )
            print(f"✅ Summary:\n{summary}")
            print(f"🧠 Reasoning:\n{log}")
        except Exception as e:
            print(f"❌ Error: {e}")
