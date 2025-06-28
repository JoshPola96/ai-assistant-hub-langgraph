import os
import re
import json
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import Union

from tools.translation_tools import deepl_translate_text
from tools.summarizer_tools import summarize_text

# --- Load Keys ---
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agent State ---
class TranslationAgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    user_input: str
    chat_history: str
    tool_output: str
    reasoning_log: List[str]
    intermediate_data: Dict[str, Any]

# --- Tool Registry ---
TOOL_REGISTRY = {
    "translation": {
        "summarize": summarize_text,
        "translate": deepl_translate_text
    }
}

# --- Utility Functions ---

def extract_last_dialogue(chat_history: str) -> str:
    pairs = re.findall(r"(User: .+?)(?:\nAgent: .+?)?", chat_history, flags=re.DOTALL)
    return pairs[-1].strip() if pairs else chat_history

def extract_json_string(text: str) -> str:
    stack = []
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append('{')
        elif char == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                return text[start:i+1]
    raise ValueError("❌ No valid JSON object found.")

def safe_translate(text: str, target_lang: str, state: TranslationAgentState) -> str:
    try:
        translated = deepl_translate_text.invoke({
            "text": text,
            "target_language": target_lang
        }).split("\n\n")[0].strip()

        if "error" in translated.lower() or "403" in translated or "Client Error" in translated:
            raise RuntimeError(translated)

        state["reasoning_log"].append("✅ Translation complete using DeepL API.")
        return translated
    except Exception as api_error:
        fallback_prompt = f"""
You are a highly intelligent multilingual assistant. Translate the following text to {target_lang} while preserving meaning, tone, and formatting.

TEXT TO TRANSLATE:
{text}
"""
        translated = llm.invoke([HumanMessage(content=fallback_prompt)]).content
        state["reasoning_log"].append(f"⚠️ DeepL failed: {api_error}")
        state["reasoning_log"].append("💡 Fallback: Translated using internal Gemma capabilities.")
        state["intermediate_data"]["fallback_translation"] = translated
        return translated

def safe_summarize(text: str, state: TranslationAgentState, key: str) -> str:
    try:
        summary = summarize_text.invoke({"text": text, "length": "short"})
        state["intermediate_data"][key] = summary
        state["reasoning_log"].append(f"🧠 Summary completed ({key.replace('_', ' ')}).")
        return summary
    except Exception as e:
        state["intermediate_data"][key] = "Failed to summarize."
        state["reasoning_log"].append(f"⚠️ Summarization failed ({key}): {e}")
        return text  # fallback: use original

# --- Core Agent Function ---
def run_translation_agent(query: str, chat_history: str = "", intermediate_data: Dict[str, Any] = {}) -> tuple[str, str]:
    state = TranslationAgentState(
        messages=[],
        user_input=query.strip(),
        chat_history=chat_history,
        tool_output="",
        reasoning_log=["[Translation Agent] 🌍 Starting multilingual assistant."],
        intermediate_data=intermediate_data or {}
    )
    state["reasoning_log"].append(f"→ User query: {query}")
    relevant_history = extract_last_dialogue(chat_history)

    # --- Planner Phase ---
    planner_prompt = f"""
    You are an elite multilingual intelligence planner inside a modular AI agent system. Your mission is to generate precise, structured translation plans from natural language requests.

    🌍 CORE MISSION:
    Transform multilingual input into culturally-aware translations that preserve tone, intent, and formatting. Support optional summarization when useful or explicitly requested.

    🛠️ AVAILABLE TOOLS:
    - DeepL Translate API (may be unavailable)
    - Summarizer (summarize before or after translation)
    - Fallback internal LLM-based translation
    - Chat history + intermediate memory data (e.g. prior outputs, uploaded content)

    📊 INTELLIGENCE CONTEXT:
    - 🗣️ Latest User Input: {state["user_input"]}
    - 🧠 Chat History (last 500 chars): {state["chat_history"][-500:] if state.get("chat_history") else "None"}
    - 🧾 Last Message: {relevant_history}
    - 📦 Memory Preview: {json.dumps({k: (str(v)[:100] + ("..." if len(str(v)) > 100 else "")) for k, v in state.get("intermediate_data", {}).items()}, indent=2) if state.get("intermediate_data") else "None"}

    ---

    🧠 STRATEGIC PROCESSING:

    **STEP 1: Identify Text Source**
    • Direct Quote → Inline string or quoted sentence
    • File Reference → Look for document or upload hints in memory
    • Chat Context → Refer to previous messages
    • Memory → Pull from intermediate_data if referenced or implied

    **STEP 2: Detect Target Language**
    • Use explicit instructions ("to French")
    • Infer from user profile, intent, or common usage
    • Default to "English" if unclear

    **STEP 3: Determine Summarization**
    • Use "before" if request says "summarize and translate"
    • Use "after" if request says "translate then summarize"
    • Use "before" if content > 500 words
    • Use false if no summarization is needed

    ---

    📤 OUTPUT FORMAT:

    Return **ONLY** a valid JSON object using this exact structure:

    {{
    "text_to_translate": "<text>",
    "target_language": "<language>",
    "summarize": "before" | "after" | false
    }}

    ⚠️ RULES:
    - DO NOT include markdown, comments, explanations, or extra keys
    - DO NOT wrap in triple backticks
    - `"summarize"` must be **exactly** `"before"`, `"after"`, or `false` — not `true` or `null`

    ---

    ✅ EXAMPLES:

    ▶️ **Direct Translation**
    Input: "Translate 'Ciao, come stai?' to English"
    Output:
    {{
    "text_to_translate": "Ciao, come stai?",
    "target_language": "English",
    "summarize": false
    }}

    ▶️ **Summarize Before Translation**
    Input: "Summarize and translate this article to French"
    Output:
    {{
    "text_to_translate": "[article_content]",
    "target_language": "French",
    "summarize": "before"
    }}

    ▶️ **Summarize After Translation**
    Input: "Translate this to Spanish and summarize it later"
    Output:
    {{
    "text_to_translate": "[extracted_text]",
    "target_language": "Spanish",
    "summarize": "after"
    }}

    ▶️ **Context Translation (Memory)**
    Input: "Convert the property details to German"
    Output:
    {{
    "text_to_translate": "[property_details_from_intermediate_data]",
    "target_language": "German",
    "summarize": false
    }}

    ▶️ **Chat Translation**
    Input: "Translate our last conversation to Hindi"
    Output:
    {{
    "text_to_translate": "[chat_history_content]",
    "target_language": "Hindi",
    "summarize": false
    }}

    ▶️ **Implicit Summarization**
    Input: "Translate this long article to Portuguese"
    (assuming it's over 500 words)
    Output:
    {{
    "text_to_translate": "[long_article_content]",
    "target_language": "Portuguese",
    "summarize": "before"
    }}

    ---

    🚀 EXECUTION INSTRUCTION:

    Analyze the inputs using your intelligence framework. Extract the source content, identify the target language, decide on summarization strategy, and return the plan using the exact JSON schema above — and nothing else.
    """

    plan_response = llm.invoke([HumanMessage(content=planner_prompt)]).content
    state["intermediate_data"]["llm_plan_raw"] = plan_response
    plan = json.loads(extract_json_string(plan_response))

    text = plan["text_to_translate"]
    target_lang = plan["target_language"]
    summarize = plan.get("summarize", False)

    state["reasoning_log"].append(f"📥 Text to translate: {text[:60]}...")
    state["reasoning_log"].append(f"🎯 Target language: {target_lang}")
    state["reasoning_log"].append(f"✂️ Summarization step: {summarize}")

    # --- Summarize before ---
    if summarize == "before":
        text = safe_summarize(text, state, key="summary_before")

    # --- Translation ---
    translated = safe_translate(text, target_lang, state)

    # --- Summarize after ---
    if summarize == "after":
        summarized = safe_summarize(translated, state, key="summary_after")
        state["tool_output"] = summarized
    else:
        state["tool_output"] = translated

    state["tool_output"] = translated
    return translated, "\n".join(state["reasoning_log"])

# --- LangGraph Node ---
def translation_agent_node(state: TranslationAgentState) -> TranslationAgentState:
    output, reasoning = run_translation_agent(
        query=state["user_input"],
        chat_history=state.get("chat_history", ""),
        intermediate_data=state.get("intermediate_data", {})
    )
    state["tool_output"] = output
    state["reasoning_log"].extend(reasoning.splitlines())
    return state

# --- (Optional) Summarizer Node ---
def summarizer_agent_node(state: TranslationAgentState) -> TranslationAgentState:
    summary = summarize_text.invoke({"text": state["user_input"], "length": "short"})
    state["tool_output"] = summary
    state["reasoning_log"].append("📄 Summary complete.")
    return state

# --- LangGraph Wrapper ---
def create_translation_graph():
    g = StateGraph(TranslationAgentState)
    g.set_entry_point("translate")
    g.add_node("translate", translation_agent_node)
    g.add_edge("translate", END)
    return g.compile()

# --- Standalone Test ---
if __name__ == "__main__":
    def test_case(title, query, chat_history=""):
        print("=" * 60)
        print(f"TEST: {title}")
        print("=" * 60)
        output, log = run_translation_agent(query, chat_history)
        print("\n--- Output ---\n", output)
        print("\n--- Reasoning Log ---\n", log)
        print("\n")

    test_case("Simple Spanish to English", "Translate 'Hola, ¿cómo estás?' to English")
    test_case("Summarize and Translate", "Translate this long article to German:\n" + ("Dies ist ein langer Artikel. "))
    test_case("Chat History Context", "Translate our last conversation to Hindi", chat_history="User: Salut!\nAgent: Bonjour!")
    test_case("Summarize After Translation","Translate this to French and summarize after:\n" + ("Ceci est un très long texte. " * 2))
    test_case("Explicit Summarize Before + Fallback","Please summarize and then translate to Spanish:\n" + ("Este es un texto muy largo. " * 2))
