# === tools/brainwonders_tools.py ===

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from utils.rag_utils import RagSystem, DEFAULT_LLM, DEFAULT_EMBEDDINGS, stringify_chat_history
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
BRAINWONDERS_DATA_DIR = "./data/brainwonders_docs"
BRAINWONDERS_PERSIST_DIRECTORY = "./data/brainwonders_db"

def parse_chat_history(chat_history: str):
    messages = []
    for line in chat_history.split("\n"):
        if line.startswith("Human:"):
            messages.append(HumanMessage(content=line[len("Human:"):].strip()))
        elif line.startswith("AI:"):
            messages.append(AIMessage(content=line[len("AI:"):].strip()))
    return messages

brainwonders_rag_system = None

@tool
def query_brainwonders_knowledge_base(query: str, chat_history: str = "", prompt: str = "") -> str:
    """
    Useful for answering Brainwonders-related queries using a RAG backend.
    Accepts an optional prompt for agent-level brain injection.
    """
    global brainwonders_rag_system
    if brainwonders_rag_system is None:
        brainwonders_rag_system = RagSystem(
            data_dir=BRAINWONDERS_DATA_DIR,
            persist_directory=BRAINWONDERS_PERSIST_DIRECTORY,
            llm_model=DEFAULT_LLM,
            embedding_model=DEFAULT_EMBEDDINGS,
            system_prompt=prompt
        )

    return brainwonders_rag_system.query(query, parse_chat_history(chat_history))
