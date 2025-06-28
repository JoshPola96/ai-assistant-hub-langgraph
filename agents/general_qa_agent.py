import re
import os
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json

from tools.web_search_tools import duckduckgo_search, fetch_web_page_content
from tools.summarizer_tools import summarize_text

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.6,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class GeneralQAAgentState(TypedDict):
    messages: Annotated[List, HumanMessage | AIMessage]
    user_input: str
    chat_history: str
    tool_output: str
    reasoning_log: List[str]
    intermediate_data: Dict[str, Any]

def run_general_qa_agent(query: str, chat_history: str = "", intermediate_data: Dict[str, Any] = {}) -> tuple[str, str]:
    state = GeneralQAAgentState(
        messages=[],
        user_input=query,
        chat_history=chat_history,
        tool_output="",
        reasoning_log=["[GeneralQA Agent] 🌐 Starting web-based question answering."],
        intermediate_data={}
    )
    state["reasoning_log"].append(f"→ User query: {query}")

    try:
        # Step 1: Refine search query
        search_planner_prompt = f"""
You are an elite search query optimizer with deep understanding of web search dynamics and information retrieval patterns.

MISSION: Transform the user's question into a laser-focused search query that maximizes relevant result discovery.

OPTIMIZATION PRINCIPLES:
• Extract core intent and key entities
• Remove conversational fluff and ambiguity  
• Include specific terms that signal authoritative sources
• Balance specificity with breadth for comprehensive coverage
• Consider temporal context when relevant

CONTEXT ANALYSIS:
User Question: "{query}"
Conversation Context (last 500 chars): {chat_history[-500:] if chat_history else 'N/A'}
Available CONTEXT & MEMORY: Preview:  {
    {k: (str(v)[:100] + ("..." if len(str(v)) > 100 else "")) for k, v in intermediate_data.items()}
}

STRATEGY:
1. Identify the primary information need
2. Extract essential keywords and entities
3. Add precision terms for better source targeting
4. Optimize for current, authoritative results

OUTPUT: Return ONLY the optimized search query - no explanations, no formatting, just the refined query string.
"""

        refined_query = llm.invoke([HumanMessage(content=search_planner_prompt)]).content.strip()
        state["reasoning_log"].append(f"🔎 Refined search query: {refined_query}")
        state["intermediate_data"]["refined_query"] = refined_query

        # Step 2: Perform web search
        search_results = duckduckgo_search.invoke({
            "query": refined_query,
            "chat_history": chat_history,
            "num_results": 5
        })
        state["reasoning_log"].append(f"🔍 Web search completed for: {refined_query}")
        state["intermediate_data"]["search_results_raw"] = search_results
        
        # ✅ New: Check for 202 or rate-limit signals
        if "202" in search_results or "rate limit" in search_results.lower():
            rate_limit_msg = "⚠️ Rate limit encountered from search service (HTTP 202). No URLs retrieved. Please try again later."
            state["reasoning_log"].append(rate_limit_msg)
            state["tool_output"] = rate_limit_msg
            return rate_limit_msg, "\n".join(state["reasoning_log"])

        urls = re.findall(r'link:\s*(https?://[^\s,]+)', search_results)
        state["intermediate_data"]["top_urls"] = urls

        if not urls:
            fallback = "I couldn’t find any relevant web results."
            state["reasoning_log"].append("⚠️ No URLs found.")
            state["tool_output"] = fallback
            return fallback, "\n".join(state["reasoning_log"])

        # Step 3: Fetch and summarize top 5 pages
        summaries, errors = [], []
        for i, url in enumerate(urls[:5], start=1):
            try:
                state["reasoning_log"].append(f"🌐 Fetching: {url}")
                page = fetch_web_page_content.invoke({"url": url})

                summary = summarize_text.invoke({
                    "text": page,
                    "length": "short"
                })
                
                summaries.append(f"🔗 {url}\n{summary}")
                state["reasoning_log"].append(f"✅ Summarized page {i}.")
                
            except Exception as e:
                err_msg = f"❌ Error processing {url}: {e}"
                summaries.append(err_msg)
                errors.append(err_msg)
                state["reasoning_log"].append(err_msg)

        state["intermediate_data"]["summaries"] = summaries
        state["intermediate_data"]["errors"] = errors
        combined_summary = "\n\n".join(summaries)

        # Step 4: Final synthesis
        synthesis_prompt = f"""
You are an expert knowledge synthesizer and communicator, capable of transforming complex web information into clear, actionable insights.

MISSION: Analyze the gathered information and craft a comprehensive, intelligent response that directly addresses the user's question.

SYNTHESIS FRAMEWORK:
🎯 ACCURACY: Base all claims on provided evidence
🔍 COMPREHENSIVENESS: Address all aspects of the question
🧠 INTELLIGENCE: Add valuable context and connections
✨ CLARITY: Use natural, engaging language
🏗️ STRUCTURE: Organize information logically

USER'S QUESTION:
{query}

INFORMATION SOURCES:
{combined_summary}

Available CONTEXT & MEMORY: Preview: { 
    {k: (str(v)[:100] + ("..." if len(str(v)) > 100 else "")) for k, v in state.get("intermediate_data", {}).items()}
}

RESPONSE STRATEGY:
1. **Direct Answer**: Lead with the core answer to the user's question
2. **Supporting Evidence**: Integrate key findings from your sources naturally
3. **Contextual Insights**: Add relevant background or implications when valuable
4. **Practical Application**: Include actionable takeaways when appropriate

QUALITY STANDARDS:
• Synthesize rather than summarize - create new understanding
• Maintain conversational tone while being authoritative  
• Avoid redundancy and filler content
• Connect information pieces to create coherent narrative
• Balance detail with readability

Generate your expert response now - be insightful, accurate, and genuinely helpful.
"""

        final_answer = llm.invoke([HumanMessage(content=synthesis_prompt)]).content.strip()
        state["reasoning_log"].append("🧠 Final answer synthesized.")

        full_response = f"""🧠 **Answer to:** _{query}_\n\n{final_answer}\n\n📚 **Sources consulted:**\n{chr(10).join([f"- {url}" for url in urls[:5]])}"""
        state["tool_output"] = full_response
        return full_response, "\n".join(state["reasoning_log"])

    except Exception as e:
        err_msg = f"❌ Error in General QA Agent: {e}"
        state["reasoning_log"].append(err_msg)
        state["tool_output"] = err_msg
        return err_msg, "\n".join(state["reasoning_log"])

if __name__ == "__main__":
    test_queries = [
        "What are the health benefits of drinking green tea?",
        "Explain the latest updates on the NEET exam eligibility criteria.",
        "Who owns OpenAI and what is its current board structure?"
    ]

    for i, q in enumerate(test_queries):
        print(f"\n{'=' * 50}")
        print(f"TEST {i + 1}: {q}")
        print(f"{'=' * 50}")
        response, reasoning = run_general_qa_agent(q)
        print(f"\n🧠 Final Answer:\n{response}")
        print(f"\n📝 Reasoning Log:\n{reasoning}")
