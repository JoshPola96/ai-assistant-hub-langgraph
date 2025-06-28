# 🧠 AIAssistantHub-LangGraph

A modular, multi-agent assistant framework powered by LangGraph, built to handle diverse and complex user queries intelligently.

This is Version 1 (v1) — more agents, refinements, and features will be added soon.

## 🚀 Overview

AIAssistantHub-LangGraph orchestrates specialized agents to solve user queries in a robust, reflective, and context-aware manner. It leverages:

  * 🤖 **Specialized agents** for real estate, summarization, RAG-based internal QA (Brainwonders), translation, general QA, and utility tasks.
  * 🔁 **Intelligent planning and fallback strategies** using LangGraph's state graph.
  * 🧠 **Prompt engineering** to simulate advanced function chaining and reasoning.

> *Please note: The free tier may experience spin-up delays during initial access after inactivity, and the project relies on free/open-source models and APIs leading to rate limits and modern LLM - LangGraph functionalities like function chaining, worked around using prompt engineering.*

## 💻 Setup

To get started with AIAssistantHub-LangGraph, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshPola96/AIAssistantHub-LangGraph.git
    ```

2.  **Navigate into the project directory:**

    ```bash
    cd AIAssistantHub-LangGraph
    ```

3.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment:**

      * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
      * On Windows:
        ```bash
        venv\Scripts\activate
        ```

5.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

6.  **Configure environment variables:**

    ```bash
    cp .env.example .env
    ```

    Edit your `.env` file to add necessary API keys (e.g., `GOOGLE_API_KEY`).

## ⚙️ Running the Application

Once setup is complete, run the Streamlit application using:

```bash
streamlit run app.py
```

## 🧩 Built-in Agents

AIAssistantHub-LangGraph comes with several specialized agents designed to handle specific types of queries:

  * **🏠 Real Estate Agent:** Handles property queries, listings, trends, and insights. It can also fetch and summarize external property-related data. This agent uses mock data located in `data/real_estate_data/`.

  * **📄 File Summarizer Agent:** Summarizes uploaded documents or entire chat histories. It supports various file types and large text summarization workflows.

  * **🧬 Brainwonders RAG Agent:** Answers internal knowledge base queries specifically about Brainwonders. This agent utilizes Retrieval-Augmented Generation (RAG) built from documents stored in `./data/brainwonders_docs`. It requires ChromaDB or similar vector storage for persistence, which is ignored from Git in `data/brainwonders_db/`.

  * **🌐 General QA Agent:** Handles factual or public knowledge questions. It searches the web, fetches relevant pages, and summarizes content as needed. This agent acts as a fallback for general topics not covered by other specialized agents.

  * **🔧 Utility Agent:** Performs a variety of utility tasks including unit conversions, timezone and datetime lookups, geocoding, IP information retrieval, weather forecasts, public holidays, jokes, quotes, and more.

  * **🌍 Translation Agent:** Translates text to user-specified languages. It has the capability to summarize content before or after translation and uses an internal LLM fallback if external translation services are unavailable or fail.

  * **💬 Gemma Fallback Agent:** Provides a last-resort direct response using the Gemma LLM. This agent is activated when all other specialized agents fail to provide a satisfactory response or encounter rate limits (e.g., DuckDuckGo 203 errors).

## 🗂 Project Structure

```
├── agents/                           # Python package for specialized AI agents
│   ├── __init__.py                   # Makes 'agents' a Python package
│   ├── brainwonders_agent.py
│   ├── general_qa_agent.py
│   ├── real_estate_agent.py
│   ├── summarizer_agent.py
│   ├── translation_agent.py
│   └── utility_agent.py
├── data/                             # Data directory
│   ├── uploads/                      # For temporary user-uploaded files
│   ├── brainwonders_docs/            # Knowledge base documents for Brainwonders RAG
│   ├── brainwonders_db/              # ChromaDB persistence for Brainwonders RAG
│   └── real_estate_data/             # Mock real estate data (e.g., real_estate_data.json)
├── tools/                            # Python package for custom tools/functions used by agents
│   ├── __init__.py                   # Makes 'tools' a Python package
│   ├── brainwonders_tools.py
│   ├── real_estate_tools.py
│   ├── summarizer_tools.py
│   ├── translation_tools.py
│   ├── utility_tools.py
│   └── web_search_tools.py
├── utils/                            # Python package for utility functions (e.g., RAG system setup)
│   ├── __init__.py                   # Makes 'utils' a Python package
│   ├── file_service.py               # File handling utilities
│   └── rag_utils.py                  # RAG system setup utilities
├── .gitignore                        # Files/directories to ignore in Git
├── app.py                            # Main Streamlit application
├── orchestrator.py                   # LangGraph orchestrator for routing queries
└── requirements.txt                  # Python dependencies
```

## 🏗 Future Plans

  * ✨ Add more specialized agents (e.g., travel advisor, calendar, etc).
  * ⚖️ Improve logic flow and structure ensuring robustness.
  * 🔗 Enhance multi-agent function chaining and memory management for more complex interactions.
  * 🌐 Integrate more robust search capabilities and external data sources.

## 🌟 Contributing

Feel free to fork the repository, open issues, or suggest improvements. This is an evolving project, and contributions are very welcome\!

## 🏁 Deployment

You can deploy this application on platforms like Render or any other service supporting Streamlit:

1.  Push your forked repository to GitHub.
2.  Connect the repository on Render and select Python as the build environment.
3.  Add your environment variables (from your `.env` file) to Render's configuration.
4.  The application will auto-deploy.

## ⚠️ Disclaimer

This is an educational project. Use at your own risk. Always validate any critical outputs, especially in domains like legal or medical.

## 💬 Contact

**Author:** Joshua Peter Polaprayil

Happy building\! 🚀✨