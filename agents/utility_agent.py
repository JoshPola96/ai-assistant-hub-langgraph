import os
import json
from typing import Dict, Any, TypedDict, Annotated, List
from decimal import Decimal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import tools.utility_tools as utility_tools
import re
import pycountry
from difflib import get_close_matches

# --- Load API key ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY missing from .env")

llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0,
    google_api_key=google_api_key,
)

# --- Define Tool Registry ---
UTILITY_REGISTRY = {
    "get_current_datetime": utility_tools.get_current_datetime,
    "convert_currency": utility_tools.convert_currency,
    "convert_units": utility_tools.convert_units,
    "get_weather": utility_tools.get_weather,
    "get_public_holidays": utility_tools.get_public_holidays,
    "lookup_ip_info": utility_tools.lookup_ip_info,
    "geocode_address": utility_tools.geocode_address,
    "reverse_geocode": utility_tools.reverse_geocode,
    "convert_timezone": utility_tools.convert_timezone,
    "get_random_joke": utility_tools.get_random_joke,
    "get_random_quote": utility_tools.get_random_quote,
}

# --- Timezone aliases ---
TIMEZONE_ALIASES = {
    "IST": "Asia/Kolkata",
    "PST": "America/Los_Angeles",
    "EST": "America/New_York",
    "CST": "America/Chicago",
    "GMT": "Etc/GMT",
    "UTC": "Etc/UTC",
}

# --- Unit Synonyms ---
UNIT_SYNONYMS = {
    "centimeters": "cm", "centimetre": "cm",
    "step": "steps",
    "meters": "m", "meter": "m", "metre": "m", 
    "feet": "ft", "ft": "ft",
    "liters": "liter", "litre": "liter", 
    "gallons": "gallon",
    "pounds": "pound", "lbs": "pound", "lb": "pound",
    "ounces": "ounce",
    "grams": "gram", 
    "kilograms": "kilogram", "kgs": "kilogram", "kg": "kilogram",
    "miles": "miles", 
    "kilometers": "kilometer", "km": "kilometer",
    "fahrenheit": "degree fahrenheit", "degrees fahrenheit": "degree fahrenheit", "degree fahrenheit": "degree fahrenheit", "°f": "degree fahrenheit" ,"°F": "degree fahrenheit",
    "celsius": "degree celsius", "degree celsius": "degree celsius", "°c": "degree celsius", "°C": "degree celsius"    
}

# --- State ---
class UtilityAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_input: str
    tool_output: str
    chat_history: str
    reasoning_log: List[str]
    intermediate_data: Dict[str, Any]

# --- Planner Node ---
def plan_and_execute(state: UtilityAgentState) -> UtilityAgentState:
    user_input = state["user_input"]
    chat_snippet = state["chat_history"][-500:] if state.get("chat_history") else "None"
    last_output = state.get("intermediate_data", {}).get("last_tool_output", "None")
    recent_reasoning = state["reasoning_log"][-1] if state.get("reasoning_log") else "None"
    timezone_aliases = json.dumps(TIMEZONE_ALIASES)
    unit_synonyms = json.dumps(UNIT_SYNONYMS)


    main_prompt = f"""
🦾 UTILITY AGENT CORE PROTOCOL v3.0

You are an elite utility specialist with multi-dimensional problem-solving capabilities. Your mission: analyze, execute, and deliver exceptional results through strategic tool orchestration.

## OPERATIONAL MANDATE
Execute with precision. Respond with pure JSON. Maximize user value through intelligent tool chaining and contextual awareness.

## CRITICAL EXECUTION RULES
```
RESPONSE_FORMAT: VALID_JSON_ONLY
MARKDOWN_ALLOWED: false
EXPLANATIONS_OUTSIDE_JSON: false
PARAMETER_MATCHING: exact_specification_required
FAILURE_RECOVERY: general_knowledge_fallback
LOOP_PREVENTION: active_monitoring
```

## COGNITIVE FRAMEWORK

### 🎯 Query Analysis Matrix
```
INTENT_DETECTION:
- Temporal → datetime/timezone tools
- Spatial → geocoding/IP location tools  
- Conversion → currency/units/timezone tools
- Information → holidays/weather tools
- Entertainment → jokes/quotes tools
- Ambiguous → clarification_required
```

### 🔄 Execution Strategy
```
SINGLE_QUERY: Direct tool execution
COMPLEX_QUERY: Tool chaining with intermediate context
FAILED_QUERY: Graceful degradation + general knowledge
UNCLEAR_QUERY: Intelligent clarification
```

## TOOL ARSENAL & PARAMETERS

### 🕐 Temporal Tools
**get_current_datetime**
- `timezone`: String (exact: "Asia/Kolkata", "America/New_York")

**convert_timezone** 
- `time_str`: String ("YYYY-MM-DD HH:MM:SS")
- `from_tz`: String (timezone identifier)
- `to_tz`: String (timezone identifier)

**get_public_holidays**
- `year`: Number (4-digit year)
- `country_code`: String (ISO: "IN", "US", "GB")

### 🌍 Spatial Tools
**get_weather**
- `city`: String (clear city name: "Mumbai", "New York")

**geocode_address**
- `address`: String (landmark/address: "Eiffel Tower", "Times Square")

**reverse_geocode**
- `lat`: Number (latitude decimal)
- `lon`: Number (longitude decimal)

**lookup_ip_info**
- `ip`: String (optional - empty for current IP)

### 🔄 Conversion Tools
**convert_currency**
- `amount`: Number (decimal allowed)
- `from_currency`: String (ISO code: "USD", "INR")
- `to_currency`: String (ISO code: "USD", "INR")

**convert_units**
- `quantity`: Number (decimal allowed)
- `from_unit`: String (see unit aliases)
- `to_unit`: String (see unit aliases)

### 🎭 Utility Tools
**get_random_joke**
- No parameters required

**get_random_quote**
- No parameters required

## CONTEXTUAL INTELLIGENCE

### 📊 Current State Analysis
```
QUERY: "{user_input}"
History (last 500 chars): "{chat_snippet}"
LAST_OUTPUT: "{last_output}"
Recent Reasoning: "{recent_reasoning}"
```

### 🧠 Decision Engine
```
IF timezone_query → resolve_timezone_aliases → get_current_datetime
IF location_query → geocode_address → get_weather 
IF conversion_query → identify_type → execute_converter
IF complex_query → chain_tools → synthesize_results
IF unclear_query → request_clarification
IF tool_failure → fallback_to_general_knowledge
```

### 🔧 Advanced Features
- **Timezone Intelligence**: {timezone_aliases}
- **Unit Flexibility**: {unit_synonyms}
- **Context Awareness**: Leverage chat history for continuity
- **Smart Chaining**: Connect related tools for complex queries
- **Error Recovery**: Graceful fallbacks with helpful alternatives

## RESPONSE PROTOCOLS

### 🎯 Success Response
```json
{{{{
  "action": "tool_call",
  "tool_name": "exact_tool_name",
  "parameters": {{{{"param": "value"}}}},
  "response": "contextual_explanation"
}}}}
```

### 💬 Direct Answer
```json
{{{{
  "action": "direct_answer", 
  "tool_name": "none",
  "parameters": {{}},
  "response": "helpful_direct_response"
}}}}
```

### ❓ Clarification Request
```json
{{{{
  "action": "clarification",
  "tool_name": "clarification", 
  "parameters": {{}},
  "response": "specific_clarification_question"
}}}}
```

## EXECUTION EXAMPLES

### Complex Query Chain
```
User: "What time is it in Tokyo when it's 3 PM in New York?"
→ convert_timezone(time_str="2024-01-01 15:00:00", from_tz="America/New_York", to_tz="Asia/Tokyo")
```

### Intelligent Clarification
```
User: "Convert 100 to euros"
→ clarification("Convert 100 of which currency to euros? Please specify the source currency.")
```

## TOOL SEQUENCING POLICY

- Use `"action": "tool_sequence"` **only** when the query explicitly requires multiple tools in a connected flow (for example: get coordinates first, then get weather).
- Otherwise, prefer a single `"tool_call"` for simpler, direct tasks.

### 💡 Example: Multi-step tool sequence

```json
{{{{
  "action": "tool_sequence",
  "tool_sequence": [
    {{{{
      "tool_name": "geocode_address",
      "parameters": {{{{"address": "Times Square"}}}}
    }}}},
    {{{{
      "tool_name": "get_weather",
      "parameters": {{{{"city": "New York"}}}}
    }}}}
  ],
  "response": "I found the coordinates for Times Square and fetched the current weather in New York for you."
}}}}

### Smart Fallback
```
Tool fails → Direct answer with general knowledge + suggestion for alternative approach
```

---
## FINAL ANSWER EVALUATION PHASE

Before finalizing your JSON response, perform a final self-check:

- ✅ Did you fully answer the user's core question?
- ✅ Is the response clear, correct, and actionable?
- ✅ Are there no unresolved ambiguities?
- ❌ If not, replan or ask for clarification instead of providing an incomplete or vague answer.

**EXECUTE WITH PURE JSON RESPONSE NOW:**
"""

    try:
        result = llm.invoke([HumanMessage(content=main_prompt)])
        response_text = result.content.strip().strip("`").strip("json")

        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}

        action = parsed.get("action", "direct_answer")
        tool_name = parsed.get("tool_name", "none")
        parameters = parsed.get("parameters", {})
        response = parsed.get("response", "")

        # Clarification or direct answer
        if action == "clarification" or tool_name == "clarification":
            msg = response or "Please provide more details."
            state["messages"].append(AIMessage(content=msg))
            state["tool_output"] = msg
            state["reasoning_log"].append("❓ Clarification requested")
            return state

        if action == "direct_answer" or tool_name == "none":
            msg = response or "I can help you with various utilities. What would you like to know?"
            state["messages"].append(AIMessage(content=msg))
            state["tool_output"] = msg
            state["reasoning_log"].append("💬 Direct answer given")
            return state

        tool_sequence = parsed.get("tool_sequence", [])
        sequence = tool_sequence if action == "tool_sequence" and tool_sequence else [
            {"tool_name": tool_name, "parameters": parameters}
        ]

        final_output = None

        for idx, step in enumerate(sequence):
            step_tool = step.get("tool_name")
            step_params = step.get("parameters", {})
            state["reasoning_log"].append(f"🔧 Step {idx+1}: {step_tool} -> {step_params}")

            if step_tool not in UTILITY_REGISTRY:
                msg = f"❌ Unknown tool in step {idx+1}: {step_tool}"
                state["messages"].append(AIMessage(content=msg))
                state["reasoning_log"].append(msg)
                return state

            processed_params = normalize_params(step_tool, step_params, state)
            if processed_params is None:
                return state  # Normalization error already handled

            try:
                result = UTILITY_REGISTRY[step_tool]() if step_tool in ["get_random_joke", "get_random_quote"] else UTILITY_REGISTRY[step_tool](**processed_params)
                state["intermediate_data"][step_tool] = result
                state["intermediate_data"]["last_tool_output"] = result
                final_output = result
                state["reasoning_log"].append(f"✅ Step {idx+1} succeeded")
            except Exception as e:
                fallback = handle_tool_failure(user_input, str(e))
                state["messages"].append(AIMessage(content=fallback))
                state["tool_output"] = fallback
                state["reasoning_log"].append(f"❌ Step {idx+1} failed: {e}")
                return state

        if final_output:
            merged_results = {
                k: v for k, v in state["intermediate_data"].items()
                if k in UTILITY_REGISTRY and not k.startswith("last_")
            }
            final_answer = respond_to_tool_outputs(user_input, merged_results, state)
            state["messages"].append(AIMessage(content=final_answer))
            state["tool_output"] = final_answer
            state["reasoning_log"].append("🧠 Final merged response generated")

        return state

    except Exception as e:
        msg = f"❌ Utility Agent failed: {e}"
        state["messages"].append(AIMessage(content=msg))
        state["tool_output"] = msg
        state["reasoning_log"].append(msg)
        return state


def normalize_params(tool_name: str, params: Dict[str, Any], state: UtilityAgentState) -> Dict[str, Any] | None:
    try:
        if tool_name in ["get_current_datetime", "convert_timezone"]:
            for k in ["timezone", "from_tz", "to_tz"]:
                if k in params and params[k].upper() in TIMEZONE_ALIASES:
                    params[k] = TIMEZONE_ALIASES[params[k].upper()]

        if tool_name == "convert_currency":
            params["amount"] = float(params["amount"])
            for key in ["from_currency", "to_currency"]:
                val = params.get(key, "").upper()
                currency = pycountry.currencies.get(alpha_3=val) or pycountry.currencies.get(name=val.title())
                if currency:
                    params[key] = currency.alpha_3
                else:
                    return suggest_alternative(state, val, "currency")

        if tool_name == "convert_units":
            params["quantity"] = float(params["quantity"])
            for key in ["from_unit", "to_unit"]:
                val = params[key].lower()
                if val in UNIT_SYNONYMS:
                    params[key] = UNIT_SYNONYMS[val]
                else:
                    return suggest_alternative(state, val, "unit")

        if tool_name == "reverse_geocode":
            params["lat"] = float(params["lat"])
            params["lon"] = float(params["lon"])

        if tool_name == "get_public_holidays":
            params["year"] = int(params["year"])
            if "country" in params and "country_code" not in params:
                params["country_code"] = pycountry.countries.get(name=params["country"]).alpha_2

        return params
    except Exception as e:
        state["messages"].append(AIMessage(content=f"❌ Parameter error: {e}"))
        return None


def suggest_alternative(state: UtilityAgentState, val: str, entity: str) -> None:
    candidates = [c.name for c in pycountry.currencies] if entity == "currency" else UNIT_SYNONYMS.keys()
    suggestions = get_close_matches(val.title(), candidates, n=1, cutoff=0.6)
    suggestion = suggestions[0] if suggestions else None
    msg = f"I didn’t recognize the {entity} '{val}'." + (f" Did you mean '{suggestion}'?" if suggestion else "")
    state["messages"].append(AIMessage(content=msg))
    state["tool_output"] = msg
    state["reasoning_log"].append(f"🔄 Suggested alternative for {entity}: '{val}' → '{suggestion or 'None'}'")
    return None


def handle_tool_failure(user_query: str, error: str) -> str:
    
    fallback_prompt = f"""
🔧 UTILITY AGENT RECOVERY MODE

**FAILURE CONTEXT:**
- Error: {error}
- Original Query: {user_query}
- Recovery Strategy: General knowledge + practical guidance

**MISSION:** Transform failure into helpful assistance.

**RESPONSE STRATEGY:**
1. Acknowledge the limitation gracefully
2. Provide general knowledge alternative
3. Suggest practical workarounds
4. Maintain helpful, solution-oriented tone

**EXECUTION:** Deliver a warm, informative response that turns the setback into value.
"""

    return llm.invoke([HumanMessage(content=fallback_prompt)]).content.strip()


def respond_to_tool_outputs(user_query: str, results: Dict[str, Any], state: UtilityAgentState) -> str:
    """
    Combines or reflects on tool outputs to generate a final user-friendly message.
    - If there's only one result, reflect on it directly.
    - If multiple, synthesize into a cohesive, conversational answer.
    """
    if not results:
        return "I didn’t get any results to work with."

    if len(results) == 1:
        result = list(results.values())[0]
        if isinstance(result, str) and len(result) < 50:
            return result.strip()

        tool_outputs = result
    else:
        tool_outputs = results
    
    if isinstance(tool_outputs, str):
        outputs_summary = tool_outputs
    else:
        outputs_summary = {
            k: (str(v)[:100] + ("..." if len(str(v)) > 100 else ""))
            for k, v in state.get("intermediate_data", {}).items()
        }
    
    result_prompt = f"""
🎨 UTILITY AGENT RESPONSE SYNTHESIZER

**INTELLIGENCE BRIEFING:**
- User Query: {user_query}
- Current Tool Outputs: {outputs_summary}

**TRANSFORMATION PROTOCOL:**

### 🧠 COGNITIVE PROCESSING
**IF SINGLE RESULT:**
- Convert technical output to natural conversation
- Add contextual insights and practical implications
- Inject appropriate personality and warmth

**IF MULTIPLE RESULTS:**
- Weave all outputs into unified narrative
- Create seamless information flow
- Generate valuable connections between data points

### 🎯 EXECUTION STANDARDS
**MANDATORY ELEMENTS:**
- Natural, conversational tone
- Subtle personality injection
- Practical context addition
- Clear, actionable information
- Appropriate humor/warmth

**FORBIDDEN ACTIONS:**
- Tool name exposure
- Technical jargon dumps
- Disconnected data presentation
- Robotic response patterns

### 🚀 ENHANCEMENT DIRECTIVES
1. **Human-Centric**: Speak as a knowledgeable friend, not a machine
2. **Value-Driven**: Every sentence should serve the user's needs
3. **Context-Aware**: Consider the user's likely follow-up questions
4. **Engagement-Focused**: Make technical information genuinely interesting

**MISSION OBJECTIVE:** Transform raw data into conversation so delightful that users forget they're interacting with an AI.

**EXECUTE CONVERSATIONAL BRILLIANCE:**
"""

    return llm.invoke([HumanMessage(content=result_prompt)]).content.strip()

# --- LangGraph Wiring ---
def create_utility_agent_graph():
    graph = StateGraph(UtilityAgentState)
    graph.set_entry_point("plan_and_execute")
    graph.add_node("plan_and_execute", plan_and_execute)
    graph.add_edge("plan_and_execute", END)
    return graph.compile()

utility_graph = create_utility_agent_graph()

def run_utility_agent(query: str, chat_history: Dict[str, Any] = None, intermediate_data: Dict[str, Any] = None) -> tuple[str, str]:
    print(f"[Utility Agent] Query: {query}")
    state = UtilityAgentState(
    messages=[],
    user_input=query,
    tool_output="",
    chat_history=chat_history or "",
    intermediate_data=intermediate_data or {},
    reasoning_log=["🧠 Utility Agent Started"]
)
    final_state = utility_graph.invoke(state)
    return (final_state.get("tool_output", "No output"),"\n".join(final_state.get("reasoning_log", [])))

__all__ = ["run_utility_agent"]

if __name__ == "__main__":
    print("\n====================")
    print("🧪 UTILITY AGENT TEST SUITE")
    print("====================")

    test_queries = [
    # # Existing core conversions
    # "Convert 10 kilometers to miles.",
    # "How many ounces in 500 grams?",
    # "Change 3 liters to gallons.",
    # "Convert 100 Fahrenheit to Celsius.",
    # "Convert 273 kelvinz to Celsius.",  # typo check
    # "What is 25 meters in feet?",
    
    # # Mixed casing, abbreviation, plural
    "convert 5 KG to lb",
    "What is 3 Feet in Meters?",
    "Change 1.5 Litres to Gallon",
    "turn 212 °F into °C",

    # Unknowns / suggestions
    "Convert 10 smoots to meters",  # fake unit
    "How many steps is 1 km?",      # ambiguous
    # "Convert 200 apples to oranges",# nonsense

    # Complex chains
    "What is 10 kilometers in miles and then into feet?",
    "Tell me where Times Square is and what time it is there.",
    "If it's 5 PM PST, what time is it in IST?",
    
    # # Currency edge
    # "Convert 100 dollars to yen",
    # "Change 300 Euros to INR",
    # "How much is 75 British Pounds in Canadian Dollars?",
    # "Convert 10 usd to inr",
    # "Turn 100 swiss francs to usd",

    # # Holidays / Weather / Joke fallback
    # "What are the holidays in the US this year?",
    # "Get me the weather in Tokyo",
    # "Tell me a joke"
]

    chat_history = ""
    intermediate_data = {}
    
    for query in test_queries:
        print("\n--------------------")
        print(f"📝 Query: {query}")
        try:
            answer, log = run_utility_agent(query, chat_history=chat_history, intermediate_data=intermediate_data)
            print(f"✅ Response:\n{answer}")
        except Exception as e:
            print(f"❌ Error: {e}")