from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

# Initialize LLM
summarizer_llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.2,
    google_api_key=google_api_key,
)

# Universal prompt — agent-aware via reasoning, no explicit agent_name
summarization_prompt = ChatPromptTemplate.from_template(
    """
🧠 UNIVERSAL SYNTHESIZER

You receive:
- User Query: {user_query}
- Content: {text}
- Desired summary length (if relevant): {length}

## INTELLIGENT MODE DECISION

Based on the user query and content, determine whether to provide:
- ✅ A direct factual answer (if the question is direct and fact-based)
- ✅ A short or detailed summary (if the user explicitly asks for a summary or the content is large)
- ✅ A short informative synthesis (if the request is ambiguous)

## EXECUTION RULES

- Always be concise and clear first.
- Add one sentence of supporting context only if strictly necessary.
- Avoid disclaimers, meta-comments, or mentions of tools or processes.
- Never say "Based on my analysis" or similar.

---

🎯 Content:
{text}

---
🔎 User Query:
{user_query}

---
✅ Final response:
    """
)

summarization_chain = summarization_prompt | summarizer_llm | StrOutputParser()

@tool
def summarize_text(text: str, user_query: str = "", length: str = "short") -> str:
    """
    Synthesizes content intelligently, deciding if it should be a direct answer or summary.
    - `text`: Content to analyze.
    - `user_query`: Original user question or request.
    - `length`: Desired summary length ("short", "medium", "long").
    """
    try:
        print(f"[SummarizerTool] Synthesizing (length: {length})...")
        return summarization_chain.invoke({
            "text": text,
            "user_query": user_query,
            "length": length
        })
    except Exception as e:
        print(f"[SummarizerTool] Error: {e}")
        return f"Sorry, I couldn't synthesize the text due to an error: {e}"

# CLI test example (optional)
if __name__ == "__main__":
    sample_text = (
        "Argentina is a country in South America. Its capital city is Buenos Aires, "
        "which is also its largest city and a major cultural and economic center."
    )
    question = "What is the capital of Argentina?"
    result = summarize_text.invoke({"text": sample_text, "user_query": question, "length": "short"})
    print("\n--- Synthesized Output ---\n", result)
