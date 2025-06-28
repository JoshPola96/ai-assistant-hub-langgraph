import os
import json
import re
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents.real_estate_agent import run_real_estate_agent
from agents.summarizer_agent import run_summarizer_agent
from agents.brainwonders_agent import run_brainwonders_agent
from agents.general_qa_agent import run_general_qa_agent
from agents.utility_agent import run_utility_agent
from agents.translation_agent import run_translation_agent

# --- Load API key ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

# --- Initialize LLM ---
orchestrator_llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0,
    google_api_key=google_api_key
)

# Fallback LLM for when search fails with 203
fallback_llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.3,
    google_api_key=google_api_key
)

# --- Corrected AGENTS_MAP ---
AGENTS_MAP = {
    "real_estate": {
        "name": "Real Estate Agent",
        "description": "Handles property queries, listings, and real estate insights. Also has access to DuckDuckGo search tool, web page fetch tool, and summarizer tool — making it a thorough agent."
    },
    "upload_summary": {
        "name": "File Summarizer Agent",
        "description": "Summarizes uploaded documents and can also summarize chat content. Tools used: file handler utility, summarizer tool."
    },
    "brainwonders": {
        "name": "Brainwonders RAG Agent",
        "description": "Answers queries about Brainwonders using context — a RAG-based retrieval QA bot. Tools used: RAG utility, brainwonders tool"
    },
    "simple": {
        "name": "General QA Agent",
        "description": "Handles general factual or knowledge-based questions. Has access to DuckDuckGo search tool, URL web content fetch tool, and summarizer tool."
    },
    "utility": {
        "name": "Utility Agent",
        "description": "Handles unit conversion, timezone/datetime queries, geocoding, IP info, jokes, quotes, weather, public holidays, and currency conversion using specialized tools."
    },
    "translate": {
        "name": "Translation Agent",
        "description": "Translates any input message from any language to a user-specified target language and optionally summarizes the message before/after translation."
    },
    "gemma_fallback": {
        "name": "Gemma Fallback Agent",
        "description": "Direct Gemma API response when web search fails with 203 monthly limit or all other steps fail to provide a satisfactory answer."
    }
}

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_input: str
    chat_history: str
    uploaded_content: str
    file_action_type: str
    steps_completed: List[str]
    final_summary: str
    task_type: str
    previous_task_type: str
    reasoning_log: str
    intermediate_data: dict
    last_answer: str

# Helper function to append agent results and update state
def _update_state_with_result(state: AgentState, result: str, reasoning: str, agent_name: str, intermediate_key: str = None, task_type: str = None):
    """Updates the agent state with the result, reasoning, and appends messages."""
    state["messages"].append(AIMessage(content=result))
    state["steps_completed"].append(agent_name)
    state["reasoning_log"] += f"\n{agent_name} reasoning:\n{reasoning}\n"
    if intermediate_key:
        state["intermediate_data"][intermediate_key] = result
    state["last_answer"] = result
    if task_type:
        state["task_type"] = task_type
    return state

# --- Tool Planner + Executor ---
def execute_planned_steps(state: AgentState) -> AgentState:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", "")
    reasoning_log = state.get("reasoning_log", "")
    intermediate_data = state.get("intermediate_data", {})
    last_answer = state.get("last_answer", "")
    uploaded_content = state.get("uploaded_content", "")
    file_action_type = state.get("file_action_type", "")

    plan_prompt = f"""
You are the core planner of a modular multi-agent assistant system. Your role is to analyze the user’s query, assess context, and determine the most efficient plan of action by assigning the best-fit agent(s) for each step. Think sequentially and strategically, leveraging available memory and reasoning.

Your output must include:
- A numbered plan of actions
- The most suitable agent (`tool_name`) for each step
- A rewritten version of the query for that agent
- A brief description of what that step will do

🧠 Available Agents:
- `real_estate`: Find or analyze real estate listings. Also fetches and summarizes external content about properties and neighborhoods.
- `simple`: General public knowledge questions. Searches the web and summarizes factual content. Use for anything not specific to Brainwonders or real estate.
- `brainwonders`: Internal knowledge-only agent. Use strictly for Brainwonders-related queries.
- `upload_summary`: Summarizes uploaded documents or chat history. Only use if file action = "summarize" or user asks for a chat summary.
- `utility`: For conversion (unit, currency, timezone), geocoding, public IPs, datetime, weather, jokes, quotes.
- `translate`: Translates text or files when a target language is explicitly given. Optionally supports summarization.
- `gemma_fallback`: **CRITICAL FALLBACK** - Use this agent ONLY when:
  * Web search returns 203 (monthly limit exceeded)
  * Multiple agents fail to provide satisfactory answers
  * All other steps fail and you need a final answer before giving up
  * The last_answer contains clear failure indicators


## 🔁 Problem-Solving Mandates
- Decompose complex or compound queries into distinct logical steps, each using its best-fit agent.
- Example: "Find 2-bedroom houses in Bangalore and rate them on amenities."
  → Step 1: Find properties (real_estate)
  → Step 2: Analyze and rate amenities (real_estate)

## 💬 Clarification Steps
- If a query is incomplete or ambiguous (e.g., missing language in a translation request), insert a `clarification` step with a direct question to the user.

🔒 Tool Assignment Rules:

- ✅ `real_estate`: Use for anything about buying, renting, filtering, or evaluating properties.
- ✅ `simple`: Use for general facts, questions, or web lookups not covered by other agents.
- ❌ `brainwonders`: Do not use for public info or general queries.
- ✅ `upload_summary`: Only when File Action Type is "summarize" or user requests a chat summary.
- ✅ `utility`: Use for conversion, location/time lookups, jokes, etc.
- ✅ `translate`: Only when:
    - User clearly says “translate to [language]”
    - Or the target language is specified in the query
    - Or the uploaded file has File Action Type “translate”
- 🚨 `gemma_fallback`: **MANDATORY USAGE** when:
    - Any agent reports "203" error or "monthly limit exceeded"
    - Multiple previous steps have failed (check failures list)
    - Last answer contains failure indicators like "I cannot help", "error", "failed"
    - No other agent can provide a satisfactory final answer


## ⚖️ Strict Tool Selection Rules
- ❌ Never route conversions, geocoding, or currency tasks, jokes, qotes, weather, or public holidays, etc to `simple` unless`utility` fails.
- ✅ Always split complex queries into smaller sub-steps, each using the correct specialized agent.
- 🔎 If any step requires data unavailable internally or a tool fails to answer the query(e.g., real-time price or detailed property info), then use `simple` only as a **final fallback**, and explain why.
- 🚨 **CRITICAL**: If you detect search limit (203) or multiple failures, immediately plan a `gemma_fallback` step as the final step.


- ⚠️ If translation is requested but the target language is unclear or missing, do not assume. Instead:
    - Do NOT make assumptions about unspecified languages. If unsure about user intent, it's better to ask clearly than to make incorrect assumptions.
    - Instead, generate a `clarification` step where the query is a question to the user.
    - Examples: "What language should I translate to?" or "Which file should I summarize?"

🧠 Memory & Reflection Intelligence:

- 📎 Check intermediate_data and chat_history first. If the question was already answered (e.g. “Why did the AI do that?”), reuse the prior result.
- ✅ If last_answer contains a valid answer that satisfies the current query, reuse it. Avoid re-executing agents unnecessarily.
- 🔁 If the last agent’s output was vague, mock, or failed, trigger a fallback plan using an alternate agent (e.g. `simple`).
- If the last_answer contains generic or unclear text (e.g. "I'm not sure", "I can't help", "Sorry, I didn't understand"), treat it as a failure and replan with a fallback agent.
- If a message in the threaded history already answers the current query (e.g. a prior translation or answer exists), reuse it rather than repeating the tool.
- Prefer the most recent AIMessage that resembles a valid final answer.

Examples of vague agent output:
- "I'm not sure how to help with that."
- "Unfortunately, I cannot complete your request."
- "Please try again."
- "No Response."

If such outputs are detected in last_answer, replan using a different tool or ask for clarification.

🧯 Redundancy Avoidance:

- 🧠 Do not re-summarize content unless explicitly requested again.
- 🧠 Do not re-translate content unless a different language is now specified.
- 🧠 Never repeat the same tool call for the same input unless a failure occurred.
- Do not plan multiple steps that perform the same operation with the same input.
- Do not reassign the same tool to the same content unless:
  - The first attempt failed, OR
  - A new output format or context is requested.

🛠 Query Rewriting:
- Clarify vague, partial, or ambiguous user queries.
- Add specific input or output goals for each step to maximize agent accuracy.
- Tailor query for the intended agent's abilities.
- For `gemma_fallback`: Include full context and specify what other agents failed to provide.

If the user asks *why* something happened, analyze prior tool calls, message flow, and any previous planning logic. Reflect intelligently on failures, misunderstandings, or missing user input.

Examples:
- If no response occurred after a translation attempt, reflect: Did the user fail to specify a target language?
- If an agent like Brainwonders was used mistakenly, explain the mismatch between query intent and agent rules.

📌 Output Format (strict):

PLAN:
1. [tool_name] → "[rewritten query for that agent]" → [brief action]
2. [tool_name] → "[rewritten query for that agent]" → [brief action]

📚 Examples (selected):

User: Translate the answer before
→ If language is missing:
PLAN:
1. clarification → "What language should I translate to?" → Ask the user to specify the target language

User: Translate the answer before to Spanish
→ PLAN:
1. translate → "Translate the last AI response to Spanish" → Translate previous result to Spanish

User: Why did the AI translate to Spanish when I didn’t specify a language?
→ PLAN:
1. direct_response → "Reflect on why the AI defaulted to Spanish earlier without being asked" → Analyze prior context and respond directly

User: Tell me where Times Square is and what time it is there
→ PLAN:
1. utility → "Get coordinates for Times Square" → Geocode Times Square
2. utility → "Get current time in New York City" → Return current datetime based on timezone

---

💬 User Input:
"{user_input}"

🧵 Chat History (truncated to 1000 chars):
"{chat_history[:1000] if chat_history else 'N/A'}"

🧵 Recent Messages (last 3):
{[m.content[:500] + ("..." if len(m.content) > 500 else "") for m in state["messages"][-3:]]}

**Available CONTEXT & MEMORY (preview):** {
    {k: (str(v)[:100] + ("..." if len(str(v)) > 100 else "")) for k, v in state.get("intermediate_data", {}).items()}
}

🧠 Last Agent Answer (first 500 chars):
"{last_answer[:500] if last_answer else 'None'}"

📄 Uploaded File Path:
"{uploaded_content if uploaded_content else 'None'}"

📂 File Action Type:
"{file_action_type if file_action_type else 'None'}"

- If a previous step failed (see failures list), do not repeat that step with the same tool.
- Instead, reassign the task to a fallback agent (e.g. switch from `translate` to `simple`).

🧯 Known Failures:
{json.dumps(state.get("intermediate_data", {}).get("failures", []), indent=2)}


🎯 Generate your plan based on this full context. Always minimize tool calls and reuse prior results when appropriate. Output only in the format:

PLAN:
1. [tool_name] → "[rewritten query]" → [brief action]

When reflecting on user confusion or internal behavior, prefer using the full threaded message list instead of just chat_history.

If the last tool failed, replan with an alternate agent. Avoid repeating failed steps.

PHASE: FINAL EVALUATION

After completing all planned steps, explicitly evaluate the final output ("last_answer") as well as the reasoning log ("reasoning_log").:

- ✅ Does it clearly and fully address the user’s main query?
- ✅ Is the core user intent fully satisfied (e.g., actual property info if requested, or direct guidance)?
- ❌ If partial or vague, trigger replanning or fallback (e.g., use a different tool, ask clarification).
- ❌ If vague, generic, error-prone, or incomplete (e.g., "I'm not sure", "No response", "Sorry"), generate a new fallback plan:
    - Use an alternate agent (e.g., `simple`)
    - Or ask the user for clarification
    - Use the `gemma_fallback` agent to generate a final answer

This phase is mandatory. You must check and act if needed. Never end without verifying answer quality.

"""

    try:
        messages = [HumanMessage(content=plan_prompt)]
        response = orchestrator_llm.invoke(messages).content.strip()
        reasoning_log += f"📋 Plan from LLM:\n{response}\n\n"
        state["intermediate_data"]["llm_plan"] = response

        # Extract steps with rewritten queries
        step_matches = re.findall(r'\d+\.\s*(\w+)\s*→\s*"([^"]+)"\s*→', response)
        valid_tools = set(AGENTS_MAP.keys())
        valid_tools.update({"clarification", "direct_response"}) # Renamed 'none' to 'direct_response'
        executed_steps = set()

        for tool_name, rewritten_query in step_matches:
            if tool_name not in valid_tools or (tool_name, rewritten_query) in executed_steps:
                reasoning_log += f"\n⚠️ Skipped unrecognized or duplicate tool step: {tool_name} → \"{rewritten_query}\"\n"
                continue

            # ✅ Skip empty rewritten queries
            if not rewritten_query.strip():
                reasoning_log += f"\n⚠️ Skipped step due to empty rewritten query: {tool_name}\n"
                continue

            # ✅ Run tool
            executed_steps.add((tool_name, rewritten_query))

            try:
                if tool_name == "real_estate":
                    result, reasoning = run_real_estate_agent(
                        query=rewritten_query,
                        chat_history=chat_history,
                        last_properties=intermediate_data.get("last_shown_properties", []),
                        intermediate_data=state["intermediate_data"]
                    )
                    state = _update_state_with_result(state, result, reasoning, "Real Estate", "real_estate_result")

                elif tool_name == "brainwonders":
                    result, reasoning = run_brainwonders_agent(
                        query=rewritten_query,
                        chat_history=chat_history
                    )
                    state = _update_state_with_result(state, result, reasoning, "Brainwonders", "brainwonders_result")

                elif tool_name == "upload_summary":
                    result, reasoning = run_summarizer_agent(
                        file_path=state.get("uploaded_content", ""),
                        chat_history=chat_history
                    )
                    state["final_summary"] = result # Specific for summarizer
                    state = _update_state_with_result(state, result, reasoning, "File Summarizer", "summary_result")

                elif tool_name == "simple":
                    result, reasoning = run_general_qa_agent(
                        query=rewritten_query,
                        chat_history=chat_history,
                        intermediate_data=state["intermediate_data"]
                    )
                    state = _update_state_with_result(state, result, reasoning, "General QA", "qa_result")

                elif tool_name == "utility":
                    result, reasoning = run_utility_agent(
                        query=rewritten_query,
                        chat_history=chat_history,
                        intermediate_data=state["intermediate_data"]
                    )
                    state = _update_state_with_result(state, result, reasoning, "Utility", "utility_result")

                elif tool_name == "translate":
                    result, reasoning = run_translation_agent(
                        query=rewritten_query,
                        chat_history=chat_history,
                        intermediate_data=state["intermediate_data"]
                    )
                    state = _update_state_with_result(state, result, reasoning, "Translation", "translation_result")

                elif tool_name == "direct_response": # Consolidated "none" and "Gemma" for direct responses
                    result = rewritten_query.strip() if rewritten_query.strip() else "Could you clarify your request?"
                    state = _update_state_with_result(state, result, "LLM answered directly without using any tool.", "Direct Answer", task_type="direct")

                elif tool_name == "clarification":
                    result = rewritten_query.strip() if rewritten_query.strip() else "Could you clarify your request?"
                    state = _update_state_with_result(state, result, "User needs to specify more information.", "Clarification Request", task_type="clarification")

                elif tool_name == "gemma_fallback": # This will be handled by the graph flow, but included for completeness if planner *forces* it
                    # This path should ideally be taken by the graph conditional edge
                    result = run_gemma_llm(rewritten_query) # Use the gemma_fallback LLM directly
                    state = _update_state_with_result(state, result, "Gemma fallback LLM provided direct answer.", "Gemma Fallback", task_type="fallback_gemma")

                else:
                    raise Exception(f"Unknown tool: {tool_name}")

            except Exception as e:
                state["intermediate_data"].setdefault("failures", []).append({
                    "tool": tool_name,
                    "query": rewritten_query,
                    "error": str(e)
                })
                err = f"❌ Error in {AGENTS_MAP.get(tool_name, {}).get('name', tool_name)}: {e}"
                state["messages"].append(AIMessage(content=err))
                reasoning_log += err + "\n"
                state["last_answer"] = err

    except Exception as e:
        err = f"❌ Error during plan execution: {e}"
        state["messages"].append(AIMessage(content=err))
        reasoning_log += err + "\n"
        state["last_answer"] = err

    state["reasoning_log"] = reasoning_log

    # Fallback if no messages generated (should ideally be caught by evaluate_final_answer now)
    if not state["messages"]:
        fallback_msg = "⚠️ I didn't produce any output in the previous step. Could you clarify or try again?"
        state["messages"].append(AIMessage(content=fallback_msg))
        state["last_answer"] = fallback_msg
        reasoning_log += "\n⚠️ Fallback: No messages were generated. Added clarification prompt.\n"
        state["reasoning_log"] = reasoning_log

    return state

def run_gemma_llm(query: str) -> str:
    # Using the predefined fallback_llm directly
    response = fallback_llm.invoke([HumanMessage(content=query)])
    return response.content.strip() if hasattr(response, "content") else str(response)

def gemma_fallback_node(state: AgentState) -> AgentState:
    query = state.get("user_input", "")
    # If there was a previous failed answer, use it in the prompt to Gemma
    if "last_answer" in state and state["last_answer"] and "error" in state["last_answer"].lower():
        query = f"The previous attempt to answer '{state['user_input']}' resulted in an error: {state['last_answer']}. Please provide a direct answer to: {state['user_input']}"
    
    fallback_answer = run_gemma_llm(query)
    state["messages"].append(AIMessage(content=fallback_answer))
    state["last_answer"] = fallback_answer
    state["reasoning_log"] += "\n✅ Final fallback executed: Gemma LLM provided direct answer.\n"
    return state

def evaluate_final_answer(state: AgentState) -> AgentState:
    last_answer = state.get("last_answer", "").lower()
    reasoning_log = state.get("reasoning_log", "").lower()

    vague_signals = [
        "i don't know",
        "no response",
        "sorry, i cannot help",
        "please clarify",
        "could you clarify",
        "i'm not sure",
        "unfortunately",
        "failed",
        "error",
        "rate limit",
        "please try again",
        "no urls retrieved",
        "too many requests",
        "duckduckgo 202",
        "duckduckgo 203"
    ]

    is_vague = (
        not last_answer.strip()
        or any(sig in last_answer for sig in vague_signals)
        or any(sig in reasoning_log for sig in vague_signals)
    )

    if is_vague:
        # Instead of directly modifying messages/last_answer here, set task_type
        # for the graph to transition to gemma_fallback_node
        state["reasoning_log"] += "\n⚠️ Final evaluation detected vague or failed answer. Triggering Gemma LLM fallback.\n"
        state["task_type"] = "fallback_gemma" # Set task_type to trigger the transition
    else:
        state["reasoning_log"] += "\n✅ Final evaluation passed: last answer is clear and complete.\n"
        state["task_type"] = "done" # Explicitly set to done if successful

    return state

def final_eval_condition(state: AgentState) -> str:
    # This condition directly uses the task_type set by evaluate_final_answer
    return state.get("task_type", "done")

# --- LangGraph Setup ---
def create_orchestrator_graph():
    g = StateGraph(AgentState)
    g.set_entry_point("execute_plan")
    g.add_node("execute_plan", execute_planned_steps)
    g.add_node("evaluate_final_answer", evaluate_final_answer)
    g.add_node("gemma_fallback", gemma_fallback_node)
    
    # After executing the plan, always evaluate the answer
    g.add_edge("execute_plan", "evaluate_final_answer")
    
    # Conditional edge from evaluate_final_answer
    g.add_conditional_edges(
        "evaluate_final_answer",
        final_eval_condition,
        {
            "fallback_gemma": "gemma_fallback",
            "done": END
        }
    )
    # The gemma_fallback node is a terminal node if it's reached
    g.add_edge("gemma_fallback", END)
    
    return g.compile()

orchestrator_graph = create_orchestrator_graph()

# --- Updated Helper function for clean UI display ---
def format_agent_display(steps_completed: List[str]) -> str:
    """Format the agent chain display for UI"""
    if not steps_completed:
        return "🧠 No Agent →"

    # Clean up agent names and create proper display
    clean_names = [step.replace(" Agent", "").strip() for step in steps_completed]

    # Create the display format
    return f"🧠 `{' → '.join(clean_names)}` Chain →"

# --- Entry Interface ---
def route_to_agent(user_input: str, chat_history: str = "", uploaded_content: str = None, file_action_type: str = "", previous_task_type: str = "") -> tuple[str, str, str, str]:
    print(f"\n--- Orchestrator received: '{user_input}' ---")
    state = AgentState(
        messages=[],
        user_input=user_input,
        chat_history=chat_history or "",
        uploaded_content=uploaded_content or "",
        file_action_type=file_action_type or "",
        steps_completed=[],
        final_summary="",
        task_type="",
        previous_task_type=previous_task_type,
        reasoning_log="",
        intermediate_data={},
        last_answer="",
    )
    try:
        result = orchestrator_graph.invoke(state)
        # Ensure that `final_msg` is correctly retrieved even if messages list is empty but last_answer is set
        final_msg = (result["messages"][-1].content if result.get("messages") else result.get("last_answer", "No response"))

        # Use the new formatting function for clean UI display
        clean_label = format_agent_display(result["steps_completed"])

        new_task_type = result["task_type"]
        log = result.get("reasoning_log", "")
        return final_msg, clean_label, new_task_type, log
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"🤖 LangGraph execution error: {e}", "🧠 `Graph Error` →", "error", str(e)

__all__ = ["route_to_agent", "AGENTS_MAP", "format_agent_display"]

if __name__ == "__main__":
    print("\n====================")
    print("🧪 TEST SUITE")
    print("====================")

    test_queries = [
        "What is 10 kilometers in miles and then into feet?",
        "Tell me where Times Square is and what time it is there.",
        "Convert 5 feet to meters and then to centimeters",
        "Find me a 2 BHK apartment in Bandra West, Mumbai under 2 crores",
        "Tell me about the real estate trends in Bangalore",
        "Translate the following to French: 'Hello, how are you?'", # Added a translation test
        "Summarize this document: It was a dark and stormy night. The old house stood silent on the hill. Rain lashed against the windows, and the wind howled like a hungry wolf. Inside, a lone figure huddled by the dying fire, lost in thought.", # Added a summarization test (simulated upload)
        "What is Brainwonders known for?", # Brainwonders test
        "Explain the concept of quantum entanglement." # Simple QA test
    ]

    chat_history = ""
    for i, query in enumerate(test_queries, 1):
        print(f"\n--------------------")
        print(f"📝 Test {i}: {query}")
        uploaded_content_for_test = ""
        file_action_type_for_test = ""
        
        # Simulate uploaded content for summarization test
        if "summarize this document" in query.lower():
            uploaded_content_for_test = "It was a dark and stormy night. The old house stood silent on the hill. Rain lashed against the windows, and the wind howled like a hungry wolf. Inside, a lone figure huddled by the dying fire, lost in thought."
            file_action_type_for_test = "summarize"

        try:
            answer, label, task_type, log = route_to_agent(
                query, 
                chat_history=chat_history, 
                uploaded_content=uploaded_content_for_test, 
                file_action_type=file_action_type_for_test
            )
            print(f"✅ {label}")
            print(f"Response: {answer}")
            print(f"Task Type: {task_type}")
            # print(f"Reasoning Log:\n{log}") # Uncomment to see the full reasoning log
            chat_history += f"User: {query}\nAssistant: {answer}\n"
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()