# tools/translation_tools.py

import os
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# Required credentials
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_ENDPOINT = "https://api-free.deepl.com/v2/translate"

if not DEEPL_API_KEY:
    raise ValueError("DEEPL_API_KEY must be set in the .env file.")

@tool
def deepl_translate_text(text: str, target_language: str) -> str:
    """
    Translates the given text to the specified target language using DeepL API.
    Auto-detects the source language.
    """
    try:
        data = {
            "auth_key": DEEPL_API_KEY,
            "text": text,
            "target_lang": target_language.upper()
        }

        response = requests.post(DEEPL_ENDPOINT, data=data)
        response.raise_for_status()
        result = response.json()

        translations = result.get("translations", [])
        if not translations:
            return "❌ No translation received."

        detected_lang = translations[0].get("detected_source_language", "unknown")
        translated_text = translations[0].get("text", "")

        return f"🌍 Translated from {detected_lang} to {target_language.upper()}:\n\n{translated_text}"

    except Exception as e:
        return f"❌ Translation error: {e}"

if __name__ == "__main__":
    # Test example
    example = deepl_translate_text.invoke({
        "text": "Bonjour tout le monde!",
        "target_language": "EN"
    })
    print(example)
