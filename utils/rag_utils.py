# === utils/rag_utils.py ===

import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

DEFAULT_LLM = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0.6, google_api_key=google_api_key)
DEFAULT_EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# --- Utility: stringify chat history ---
def stringify_chat_history(messages):
    return "\n".join([f"{m.type.title()}: {m.content}" for m in messages])

def load_and_split_documents(data_dir: str, chunk_size: int = 800, chunk_overlap: int = 200):
    all_documents = []
    md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader)
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)

    all_documents.extend(md_loader.load())
    all_documents.extend(txt_loader.load())
    all_documents.extend(pdf_loader.load())

    if not all_documents:
        print(f"Warning: No documents found in '{data_dir}'.")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_documents)

def get_vector_store(chunks: list, persist_directory: str, embeddings: Embeddings = DEFAULT_EMBEDDINGS):
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

def create_rag_chain(retriever, llm_instance: BaseChatModel, system_prompt: str):
    prompt = ChatPromptTemplate.from_template(system_prompt)
    document_chain = create_stuff_documents_chain(llm_instance, prompt)
    return create_retrieval_chain(retriever, document_chain)

class RagSystem:
    def __init__(
        self,
        data_dir: str,
        persist_directory: str,
        llm_model: BaseChatModel = DEFAULT_LLM,
        embedding_model: Embeddings = DEFAULT_EMBEDDINGS,
        system_prompt: str = "",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        search_kwargs: dict = {"k": 3}
    ):
        self.system_prompt = system_prompt
        self.retriever = get_vector_store(
            load_and_split_documents(data_dir, chunk_size, chunk_overlap),
            persist_directory,
            embedding_model
        ).as_retriever(**search_kwargs)

        self.qa_chain = create_rag_chain(self.retriever, llm_model, self.system_prompt)

    def query(self, user_query: str, chat_history_messages: list = None) -> str:
        if chat_history_messages is None:
            chat_history_messages = []
        stringified_history = stringify_chat_history(chat_history_messages)
        try:
            result = self.qa_chain.invoke({"input": user_query, "chat_history": stringified_history})
            return result["answer"]
        except Exception as e:
            return f"Error: {e}"