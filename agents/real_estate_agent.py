# real_estate_agent.py

import os
import json
import re
from typing import Dict, Any, List, TypedDict, Optional, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.real_estate_tools import get_mock_property_listings
from tools.web_search_tools import duckduckgo_search, fetch_web_page_content
from tools.summarizer_tools import summarize_text

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment")

llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

# Tool Registry - centralized tool management
TOOL_REGISTRY = {
    "real_estate": {
        "get_listings": get_mock_property_listings,
        "search": duckduckgo_search,
        "fetch_page": fetch_web_page_content,
        "summarize": summarize_text
    }
}

# Flatten all tools for executor
TOOLS = list(TOOL_REGISTRY["real_estate"].values())
TOOL_MAP = {tool.name: tool for tool in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    user_query: str
    chat_history: str
    steps_completed: List[str]
    tool_to_execute: str
    tool_args_to_execute: Dict[str, Any]
    last_tool_calls: List[Dict[str, Any]]
    intermediate_data: Dict[str, Any]
    step_count: int
    has_executed_tool: bool
    search_urls: List[str]
    last_shown_properties: List[Dict[str, Any]]
    reasoning_log: List[str]
    last_answer: str


class SimpleToolExecutor:
    def __init__(self, tool_registry: Dict[str, Dict[str, Any]]):
        # Flatten all tools from registry
        self.tools = {}
        for agent_tools in tool_registry.values():
            for tool_name, tool_func in agent_tools.items():
                self.tools[tool_func.name] = tool_func

    def invoke(self, name: str, inputs: Dict[str, Any]):
        if "chat_history" in inputs and not isinstance(inputs["chat_history"], str):
            inputs["chat_history"] = str(inputs["chat_history"])
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        return tool.invoke(inputs)

executor = SimpleToolExecutor(TOOL_REGISTRY)
MAX_STEPS = 10

def planner(state: AgentState) -> AgentState:
    if state.get("has_executed_tool") or state["step_count"] >= MAX_STEPS:
        return state

    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        state["messages"].append(HumanMessage(content=state["user_query"]))
        
    state["reasoning_log"].append(f"🤔 Planning action for query: '{state['user_query']}'")

    # Check if we've already failed this tool recently to avoid loops
    recent_failures = [log for log in state["reasoning_log"] if "failed" in log.lower()]
    if len(recent_failures) >= 2:
        # Too many failures, provide direct answer or clarification
        fallback_msg = "I'm having trouble processing your request. Could you please specify the city/location and your preferences more clearly? For example: 'Find 2-3 BHK apartments in Mumbai under 80 lakhs'."
        state["messages"].append(AIMessage(content=fallback_msg))
        state["has_executed_tool"] = True
        state["reasoning_log"].append("🛑 Too many failures, providing fallback response")
        return state

    # Generate tool descriptions from registry
    available_tools = []
    for agent_name, agent_tools in TOOL_REGISTRY.items():
        for tool_key, tool_func in agent_tools.items():
            available_tools.append(f"- `{tool_func.name}`: {tool_func.description}")
    available_tools_str = "\n".join(available_tools)

    memory_info = ""
    if state["last_shown_properties"]:
        memory_info += "Previously shown properties:\n"
        for p in state["last_shown_properties"][:3]:
            memory_info += f"- {p.get('name', 'Property')} in {p.get('location')} | {p.get('bedrooms')} BHK | {p.get('price')}\n"
    else:
        memory_info = "None yet."

    prompt = f"""
You are an elite Real Estate Intelligence Agent with advanced reasoning, replanning capabilities, and robust error recovery. Your mission is to provide maximum value to users even when tools fail or return incomplete data.

🧠 CORE INTELLIGENCE PRINCIPLES:
- **Adaptive Planning**: If Plan A fails, automatically try Plan B, C, or D
- **Context Mastery**: Leverage all available information including memory, chat history, and partial results
- **Smart Degradation**: Always provide value, even with limited data
- **Error Recovery**: Transform failures into opportunities for better service

⚡ AVAILABLE TOOLS:

**get_mock_property_listings** — Primary property database
- Use for: Finding properties, getting property details
- Required: location (city name)
- Optional: property_type, price_min_lakhs, price_max_lakhs, min_bedrooms, max_bedrooms, amenities, sort_by, sort_order

⚙️ TOOL INSTRUCTION: get_mock_property_listings

**PRIMARY TOOL – Must always be tried first for supported queries.**

✅ REQUIRED FIELD:
• `location` (str) — e.g., "Delhi", "Bangalore"

✅ OPTIONAL FILTERS (Only include when user query mentions them):
• `property_type` (str) — one of "apartment", "villa", "plot", "any" (default)
• `price_min_lakhs` (float) — must be numeric, in lakhs (e.g., 50.0)
• `price_max_lakhs` (float) — must be numeric, in lakhs (e.g., 90.0)
• `min_bedrooms` (int)
• `max_bedrooms` (int)
• `amenities` (List[str]) — must be a **valid list of lowercase strings** (e.g., `["gym", "pool"]`). ❌ Never pass a single string.
    - ❌ Reject unsupported values like `"dogs allowed"` → fallback to web search
• `sort_by` (str) — one of: `"price"`, `"area"`, `"bedrooms"` only.
    - If user says “rooms”, interpret as `"bedrooms"`
• `sort_order` (str) — either `"asc"` or `"desc"`
• `return_format` (str) — always `"dict"`

🚫 STRICT RULES:
• Never pass non-numeric values to float fields (`price_min_lakhs`, `price_max_lakhs`)
• Never pass `sort_by` values not in the allowed list
• Never pass `amenities` as a string — must be a list
• If unsupported amenities or filters (e.g., `"dogs allowed"`), do not call this tool — use `duckduckgo_search` instead

You can interpret the user queries and rewrite it properly for the above tools based on their use and parameters.

**duckduckgo_search** — Web search for market data and property information
- Use for: Market trends, area information, property policies, builder details
- Strategy: Search for general information, not specific property IDs

**fetch_web_page_content** — Extract detailed information from web pages
**summarize_text** — Condense large amounts of information

🧠 INTELLIGENT REPLANNING FRAMEWORK:

**SCENARIO 1: Pet-Friendly Properties Query**
❌ Wrong approach: Search for "mock-del-001 pet friendly"
✅ Smart approach: 
1. First, check if property data includes pet policies
2. If not available, search for: "location pet friendly apartments" or "builder_name pet policy"
3. If still no results, provide general guidance: "Most property_type in location allow pets, but I recommend confirming with the builder/society directly"
4. Offer to help contact the properties for confirmation

**SCENARIO 2: Tool Failure Recovery**
When a tool fails or returns "no relevant information":
1. **Analyze why it failed**: Wrong search terms? No data available? Technical issue?
2. **Try alternative approach**: Different search terms, different tool, or use available context
3. **Provide partial value**: Use what you know + educate the user
4. **Offer next steps**: Suggest how they can get the missing information

**SCENARIO 3: Incomplete Data**
When you have some but not all requested information:
1. **Provide what you have**: Share available details clearly
2. **Acknowledge gaps**: Be transparent about missing information
3. **Suggest alternatives**: Offer related information or next steps
4. **Proactive help**: Anticipate follow-up questions

---

💡 CURRENT CONTEXT:

**User Query:** "{state['user_query']}"
**Chat History (last 500 chars):** "{state.get('chat_history', '')[-500:] if state.get('chat_history') else 'None'}"
**Previous Properties (keys only):** "{list(state['last_shown_properties'].keys()) if state.get('last_shown_properties') else 'None'}"
**Recent Actions (last 3):** "{', '.join(state['steps_completed'][-3:]) if state.get('steps_completed') else 'None'}"
**Previous Reasoning (last 1):** "{state['reasoning_log'][-1] if state.get('reasoning_log') else 'None'}"
**Available CONTEXT & MEMORY (preview):** { 
    {k: (str(v)[:100] + ("..." if len(str(v)) > 100 else "")) for k, v in state.get("intermediate_data", {}).items()}
}

---

🎯 SMART EXECUTION EXAMPLES:

**Pet-friendly properties with replanning:**
{{
"steps": [
    {{
    "tool_name": "duckduckgo_search",
    "tool_input": {{
        "query": "Delhi apartments pet friendly policy 2025"
    }},
    "reasoning": "Searching for general pet policies in Delhi apartments since specific property data doesn't include pet information",
    "fallback_plan": "If no results, provide general guidance and offer to help contact properties directly",
    "answer": null
    }}
]
}}

**Market information with multiple approaches:**
{{
"steps": [
    {{
    "tool_name": "duckduckgo_search",
    "tool_input": {{
        "query": "Bangalore real estate market trends 2025 investment areas"
    }},
    "reasoning": "Primary search for market trends",
    "fallback_plan": "If insufficient, search for specific areas like Whitefield, Electronic City trends",
    "answer": null
    }}
]
}}

**Using memory for property-specific queries:**
{{
"steps": [
    {{
    "tool_name": "none",
    "tool_input": {{}},
    "reasoning": "User asking about previously shown properties - using memory context",
    "answer": "Looking at the properties I showed you earlier:\n\n**Pet Policy Status:**\n• Farmhouse Retreat Chattarpur (5 BHK) - Independent property, likely pet-friendly\n• Vasant Kunj Heights (4 BHK) - Apartment complex, need to verify\n• Greater Kailash Residency (3 BHK) - Apartment, typically have pet policies\n\nFor apartments, most Delhi societies allow pets but have specific rules. I recommend checking with each society directly. Would you like me to help you find contact information for these properties?"
    }}
]
}}

**Intelligent fallback when tools fail:**
{{
"steps": [
    {{
    "tool_name": "none",
    "tool_input": {{}},
    "reasoning": "Previous search failed, providing useful guidance based on general knowledge",
    "answer": "While I couldn't find specific pet policies for these properties online, here's what I can tell you:\n\n**General Pet Policy Trends in Delhi:**\n• Independent houses/farmhouses: Usually pet-friendly\n• Apartment complexes: Most allow pets with society approval\n• Luxury complexes: Often have dedicated pet amenities\n\n**Next Steps:**\n1. I can help you get contact details for each property\n2. Call and ask about their pet policy specifically\n3. Ask about pet deposits and any restrictions\n\nWhich properties are you most interested in? I can prioritize getting their contact information."
    }}
]
}}

---

🧠 ADVANCED REASONING PROTOCOL:

**Step 1: Analyze Query Intelligence**
- What is the user actually asking for?
- What information do I already have?
- What's the best approach given available tools and context?

**Step 2: Plan with Fallbacks**
- Primary approach: Most direct tool/method
- Secondary approach: Alternative if primary fails
- Tertiary approach: Use available context + general knowledge
- Final fallback: Acknowledge limitation but provide maximum value

**Step 3: Execute with Monitoring**
- Monitor tool results quality
- If result is poor/irrelevant, trigger replanning
- Always extract maximum value from any result

**Step 4: Value Delivery**
- Even with failures, provide actionable information
- Anticipate follow-up questions
- Offer concrete next steps

---

📤 RESPONSE FORMAT (JSON only):

{{
"steps": [
    {{
    "tool_name": "<tool_name>" | "none",
    "tool_input": {{ ... }},
    "reasoning": "<why_this_approach>",
    "fallback_plan": "<what_if_this_fails>",
    "answer": "<direct_response>" | null
    }}
]
}}

🎯 EXECUTION PRIORITIES:
1. **Be Relentlessly Helpful**: Never give up on providing value
2. **Think Multiple Steps Ahead**: Plan for likely failures and alternatives
3. **Leverage All Available Context**: Memory, history, partial data
4. **Educate and Guide**: Help users understand the domain
5. **Be Proactive**: Anticipate needs and offer additional help

---

💬 FINAL ANSWER EVALUATION PHASE

Before completing your JSON plan, **critically evaluate your planned final answer**:

- ✅ Does it clearly and fully address the user’s main query?
- ✅ Is the core user intent fully satisfied (e.g., actual property info if requested, or direct guidance)?
- ✅ Is it clear, actionable, and transparent about limitations?
- ❌ If partial or vague, trigger replanning or fallback (e.g., use a different tool, ask clarification).

If the answer is clear and satisfactory, proceed.

If not, replan before finalizing the steps.

---
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]

        parsed = json.loads(response)
        if parsed.get("steps"):
            first_step = parsed["steps"][0]
            tool_name = first_step.get("tool_name", "none")
            tool_input = first_step.get("tool_input", {})
            answer = first_step.get("answer")
        
        state["reasoning_log"].append(f"🎯 Planner decided: tool='{tool_name}'")

        state["tool_to_execute"] = tool_name
        state["tool_args_to_execute"] = tool_input

        if answer and answer.strip().lower() not in ("null", "none", ""):
            state["last_answer"] = answer
            state["messages"].append(AIMessage(content=answer))
            state["has_executed_tool"] = True
            
    except Exception as e:
        print("Planner error:", e)
        # Smarter fallback based on query content
        query_lower = state["user_query"].lower()
        if any(city in query_lower for city in ["delhi", "mumbai", "bangalore", "chennai", "hyderabad", "pune"]):
            fallback_msg = "I understand you're looking for properties. Let me search for available listings."
            state["tool_to_execute"] = "get_mock_property_listings" 
            # Extract city name
            for city in ["delhi", "mumbai", "bangalore", "chennai", "hyderabad", "pune"]:
                if city in query_lower:
                    state["tool_args_to_execute"] = {"location": city.title()}
                    break
        else:
            fallback_msg = "Could you please specify which city you're interested in? I can help you find properties in major Indian cities."
            state["messages"].append(AIMessage(content=fallback_msg))
            state["has_executed_tool"] = True
        
        state["reasoning_log"].append("🔧 Used intelligent fallback")

    return state

def executor_node(state: AgentState) -> AgentState:
    tool = state["tool_to_execute"]
    args = state["tool_args_to_execute"]
    state["step_count"] += 1

    if not tool or tool == "none":
        return state

    try:
        result = executor.invoke(tool, args)
        state["steps_completed"].append(tool)
        state["intermediate_data"][tool] = result

        if tool == "get_mock_property_listings":
            # Handle both old dict format and new list format
            if isinstance(result, dict) and result.get("status") == "no_results":
                # No results found
                answer_msg = f"🔍 {result.get('message', 'No properties found.')} Would you like me to search in a different city or adjust your criteria?"
                state["messages"].append(AIMessage(content=answer_msg))
                state["last_answer"] = answer_msg
                state["reasoning_log"].append("ℹ️ No results from property search")
                
            elif isinstance(result, list) and result:
                # Process successful results
                query = state["user_query"].lower()
                
                # Smart sorting based on query intent
                if "expensive" in query or "highest" in query or "costly" in query:
                    # Sort by price descending
                    try:
                        result = sorted(result, key=lambda x: float(x.get("price", "0").replace("lakhs", "").replace("lakh", "").replace("cr", "").replace("crore", "").strip()) * (100 if "cr" in x.get("price", "") else 1), reverse=True)
                    except:
                        pass
                elif "cheap" in query or "lowest" in query or "affordable" in query:
                    # Sort by price ascending
                    try:
                        result = sorted(result, key=lambda x: float(x.get("price", "0").replace("lakhs", "").replace("lakh", "").replace("cr", "").replace("crore", "").strip()) * (100 if "cr" in x.get("price", "") else 1))
                    except:
                        pass
                elif "room" in query or "bedroom" in query:
                    # Sort by bedrooms descending
                    try:
                        result = sorted(result, key=lambda x: int(x.get("bedrooms", 0)), reverse=True)
                    except:
                        pass

                # Save to memory for future reference
                state["last_shown_properties"] = result[:10]

                # Create engaging output
                listings_md = []
                for i, r in enumerate(result[:5], 1):
                    amenities_str = ', '.join(r.get('amenities', [])[:3]) or 'Basic amenities'
                    listings_md.append(
                        f"**#{i}. {r.get('name', 'Property')}** (ID: `{r.get('_id')}`)\n"
                        f"📍 **Location:** {r.get('location')}\n"  
                        f"💰 **Price:** {r.get('price')}\n"
                        f"🛏️ **Bedrooms:** {r.get('bedrooms')} BHK\n"
                        f"📐 **Area:** {r.get('area_sqft')} sqft\n"
                        f"✨ **Amenities:** {amenities_str}"
                    )

                total_count = len(result)
                answer_msg = f"🏠 **Found {total_count} properties for you:**\n\n" + "\n\n".join(listings_md)
                
                if total_count > 5:
                    answer_msg += f"\n\n💡 *Showing top 5 results. Total {total_count} properties found.*"
                
                state["messages"].append(AIMessage(content=answer_msg))
                state["last_answer"] = answer_msg
                state["reasoning_log"].append(f"✅ Found {total_count} properties")
                
            else:
                # Handle unexpected result format
                answer_msg = "🔍 I found some properties but couldn't format them properly. Could you please try your search again?"
                state["messages"].append(AIMessage(content=answer_msg))
                state["last_answer"] = answer_msg
                state["reasoning_log"].append("⚠️ Unexpected result format")

        elif tool == "duckduckgo_search":
            if isinstance(result, str) and result.strip():
                # Extract URLs for further processing
                urls = re.findall(r'https?://[^\s,\]]+', result)
                state["search_urls"] = urls[:3]
                
                if urls:
                    answer_msg = "🔍 Found some relevant information online. Let me get the details..."
                    state["messages"].append(AIMessage(content=answer_msg))
                    state["last_answer"] = answer_msg
                    state["tool_to_execute"] = "fetch_web_page_content"
                    state["tool_args_to_execute"] = {"url": urls[0]}
                    state["has_executed_tool"] = False
                    return state
                else:
                    # Provide search results directly if no URLs
                    answer_msg = f"🔍 **Search Results:**\n\n{result[:1000]}..."
                    state["messages"].append(AIMessage(content=answer_msg))
                    state["last_answer"] = answer_msg
            else:
                answer_msg = "🔍 I couldn't find relevant information online for your query. Could you be more specific?"
                state["messages"].append(AIMessage(content=answer_msg))
                state["last_answer"] = answer_msg

        elif tool == "fetch_web_page_content":
            # Continue fetching remaining URLs or move to summarization
            remaining = state["search_urls"][1:] if state["search_urls"] else []
            if remaining:
                state["search_urls"] = remaining
                state["tool_to_execute"] = "fetch_web_page_content"
                state["tool_args_to_execute"] = {"url": remaining[0]}
                state["has_executed_tool"] = False
                return state
            else:
                # All URLs fetched, now summarize
                all_content = "\n\n".join(str(c) for c in state["intermediate_data"].values() if isinstance(c, str))
                if all_content.strip():
                    state["tool_to_execute"] = "summarize_text"
                    state["tool_args_to_execute"] = {"text": all_content[:15000], "length": "medium"}
                    state["has_executed_tool"] = False
                    return state
                else:
                    answer_msg = "🔍 I couldn't extract useful information from the web pages. Please try a different search."
                    state["messages"].append(AIMessage(content=answer_msg))
                    state["last_answer"] = answer_msg

        elif tool == "summarize_text":
            summary = result if isinstance(result, str) else str(result)
            answer_msg = f"📊 **Here's what I found:**\n\n{summary}"
            state["messages"].append(AIMessage(content=answer_msg))
            state["last_answer"] = answer_msg
            state["reasoning_log"].append("📋 Summary generated")

        state["has_executed_tool"] = True

    except Exception as e:
        error_msg = str(e)
        print(f"[Executor Error] Tool '{tool}' failed: {error_msg}")
        
        # Provide intelligent error handling
        if "could not convert string to float" in error_msg:
            answer_msg = "🔧 I encountered a data formatting issue. Let me help you differently - could you specify your budget range in lakhs? (e.g., '50-80 lakhs')"
        elif "no_results" in error_msg.lower():
            answer_msg = "🔍 No properties found matching your criteria. Would you like to try a different city or adjust your requirements?"
        else:
            answer_msg = f"⚠️ I encountered an issue while searching. Could you please rephrase your request? I can help you find properties in major Indian cities."
        
        state["messages"].append(AIMessage(content=answer_msg))
        state["last_answer"] = answer_msg
        state["reasoning_log"].append(f"❌ Tool '{tool}' failed: {error_msg[:100]}")
        state["has_executed_tool"] = True

    state["tool_to_execute"] = "none"
    state["tool_args_to_execute"] = {}
    return state

def should_continue(state: AgentState) -> str:
    try:
        if state["step_count"] >= MAX_STEPS:
            state["reasoning_log"].append(f"🛑 Reached max steps ({MAX_STEPS})")
            return END
        if state["has_executed_tool"]:
            state["reasoning_log"].append("✅ Task completed")
            return END
        if state["tool_to_execute"] != "none":
            return "executor"
        return "planner"
    except Exception as e:
        print("[Edge Decision Error]", e)
        return END

def build_real_estate_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", should_continue, {"executor": "executor", "planner": "planner", END: END})
    graph.add_conditional_edges("executor", should_continue, {"executor": "executor", "planner": "planner", END: END})
    return graph.compile()

def run_real_estate_agent(
    query: str,
    chat_history: str = "",
    last_properties: Optional[List[Dict[str, Any]]] = None,
    intermediate_data: Optional[Dict[str, Any]] = None
) -> tuple[str, str]:
    print(f"\n[Real Estate Agent] Query: {query}")
    graph = build_real_estate_graph()

    initial_state = AgentState(
        user_query=query,
        chat_history=chat_history,
        messages=[],
        steps_completed=[],
        tool_to_execute="none",
        tool_args_to_execute={},
        last_tool_calls=[],
        intermediate_data=intermediate_data or {},
        step_count=0,
        has_executed_tool=False,
        search_urls=[],
        last_shown_properties=last_properties or [],
        reasoning_log=[],
        last_answer=""
    )

    # Add initial reasoning
    initial_state["reasoning_log"].append("🚀 Real Estate Agent started")
    if chat_history:
        initial_state["reasoning_log"].append("📚 Using chat history context")
    if last_properties:
        initial_state["reasoning_log"].append(f"📦 Loaded {len(last_properties)} previous properties")

    try:
        final_state = graph.invoke(initial_state)

    # Extract final response
        messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
        final_response = messages[-1].content if messages else "Let me know how I can assist with your real estate query."
        
        # Final reasoning log
        log = final_state["reasoning_log"]
        if final_state["steps_completed"]:
            log.append(f"🏁 Agent used: {', '.join(final_state['steps_completed'])}")
        reasoning_log = "\n".join(log)
                
        return final_response, reasoning_log
    
    except Exception as e:
        error_response = f"❌ Real Estate Agent encountered an error: {str(e)}"
        error_log = f"🚀 Real Estate Agent started\n❌ Fatal error: {str(e)}"
        return error_response, error_log

if __name__ == "__main__":
    # Test the agent
    
    prior_properties = [
        {"_id": "P001", "name": "Sunrise Apartments", "location": "Whitefield", "bedrooms": 2, "price": "80 lakhs", "area_sqft": 1200, "amenities": ["Gym", "Parking"]},
        {"_id": "P002", "name": "Tech Vista", "location": "Whitefield", "bedrooms": 3, "price": "1.2 crores", "area_sqft": 1500, "amenities": ["Security", "Lift"]},
        {"_id": "P003", "name": "Lakeview Residency", "location": "Whitefield", "bedrooms": 2, "price": "70 lakhs", "area_sqft": 1100, "amenities": ["Clubhouse", "Pool"]},
    ]
        
    test_queries = [
        "Find a 2 BHK in Mumbai under 50 lakhs",
        "Find me houses in bangalore with 2 bedrooms",
        "I'm looking to buy a house in Bangalore with amentities like gym",
        "What are the best areas to invest in Bangalore?",
        "Tell me about the second property",
        "Can you give me a summary of the first property?",
        "Find a two bedroom property near the headquarters of Brainwonders.",
        "What would you do if a tool fails or the returned answer isnt exactly relevant?",
        "What's the outlook for the Hyderabad property market in 2025?",
        "I'm looking to buy"
        ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}: {query}")
        print('='*50)
        
        response, reasoning = run_real_estate_agent(query)
        print(f"\nReasoning:\n{reasoning}")
        print(f"\nResponse:\n{response}")
        
        