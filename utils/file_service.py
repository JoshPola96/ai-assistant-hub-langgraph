# tools/file_service.py
from langchain_core.tools import tool
import os

@tool
def read_uploaded_file(file_path: str) -> str:
    """
    Reads the content of a text or PDF file from the given path.
    Supports .txt and .pdf formats.
    Returns plain text or an error message.
    """
    try:
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_path.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text.strip() else "No readable text found in the PDF."

        else:
            return "Unsupported file type. Please upload a .txt or .pdf file."

    except Exception as e:
        return f"Failed to read file: {e}"
