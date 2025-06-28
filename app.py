import os
import streamlit as st
from orchestrator import route_to_agent, AGENTS_MAP
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Assistant Hub", layout="wide", page_icon="🤖")
st.title("🤖 AI Assistant Hub")
st.caption("A modular, multi-agent assistant system.")

# Show available tools
with st.expander("⚙️ Available Tools", expanded=False):
    for tool in AGENTS_MAP.values():
        st.markdown(f"**`{tool['name']}`**: {tool['description']}")

# Ensure upload folder exists
os.makedirs("./data/uploads", exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hello! How can I help you today?"}]
if "file_uploader_key_counter" not in st.session_state:
    st.session_state.file_uploader_key_counter = 0
if "processing_file" not in st.session_state:
    st.session_state.processing_file = False

# Show chat history
for msg in st.session_state.messages:
    role, content = msg["role"], msg["content"]
    st.chat_message(role).write(content)
    
file_action = None
target_language = None

# File upload section
uploaded_file = st.sidebar.file_uploader(
    "Upload a document to process",
    type=["txt", "csv", "json", "xml", "log", "pdf"],
    key=f"file_uploader_{st.session_state.file_uploader_key_counter}"
)

if uploaded_file:
    # Show dropdown to select action
    file_action = st.sidebar.selectbox(
        "What would you like to do with this file?",
        ["Summarize", "Translate"],
        key="file_action"
    )
    
# Let user specify target language if translating
if file_action == "Translate":
    target_language = st.sidebar.text_input("Target Language (e.g., Spanish, French)", value="", key="target_language")

# Let user confirm
if st.sidebar.button("🚀 Process File"):
    st.session_state.processing_file = True
    file_path = f"./data/uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Add message to chat
    st.chat_message("user").write(f"📄 Uploaded file: `{uploaded_file.name}` ({file_action})")
    st.session_state.messages.append({
        "role": "user",
        "content": f"📄 Uploaded file: `{uploaded_file.name}` ({file_action})"
    })

    if file_action == "Translate" and target_language:
        generic_file_prompt = f"Please translate the uploaded file to {target_language}: {uploaded_file.name}"
    else:
        generic_file_prompt = f"Please {file_action.lower()} the uploaded file: {uploaded_file.name}"

    with st.spinner(f"Processing '{uploaded_file.name}' via Orchestrator..."):
            try:
                response, label, new_task_type, reasoning = route_to_agent(
                    user_input=generic_file_prompt,
                    chat_history="\n".join([
                        f"{m['role']}: {m['content']}" for m in st.session_state.messages
                    ]),
                    uploaded_content=file_path,
                    file_action_type=file_action.lower()
                )

                # Reasoning
                if label and label != "Direct Response":
                    steps_inline = " → ".join([
                        line.strip().replace("➡️", "").strip()
                        for line in reasoning.strip().splitlines()
                        if line.strip()
                    ])
                    reasoning_msg = f"🧠 `{label}` Agent → {steps_inline}"
                    st.chat_message("ai").write(reasoning_msg)
                    st.session_state.messages.append({"role": "ai", "content": reasoning_msg})

                st.chat_message("ai").write(response)
                st.session_state.messages.append({"role": "ai", "content": response})
                
                if (
                    new_task_type == "upload_summary" 
                    or "Summarizer" in label 
                    or "Translation" in label
                ):
                    st.session_state.latest_file_content = response
                    
                    if "Translation" in label:
                        st.session_state.latest_file_type = "translation"
                        st.session_state.latest_file_name = f"translated_{uploaded_file.name}.txt"
                        st.session_state.latest_file_label = "Download Translation"
                    elif "Summarizer" in label:
                        st.session_state.latest_file_type = "summary"
                        st.session_state.latest_file_name = f"summary_{uploaded_file.name}.txt"
                        st.session_state.latest_file_label = "Download Summary"
                    else:
                        st.session_state.latest_file_type = "file"
                        st.session_state.latest_file_name = f"processed_{uploaded_file.name}.txt"
                        st.session_state.latest_file_label = "Download File"            
                    st.session_state.latest_summary = response    
                    
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
            finally:
                st.session_state.processing_file = False
                st.session_state.file_uploader_key_counter += 1
            st.rerun()

# Chat input handling
if prompt := st.chat_input("Ask me something!"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    full_history = "\n".join([
        f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in st.session_state.messages if m["role"] in ["user", "ai"]
    ])

    with st.spinner("Thinking..."):
        try:
            response, label, new_task_type, reasoning_log = route_to_agent(prompt, full_history)

            if label and label != "Direct Response":
                # Combine tool label and reasoning
                steps_inline = " → ".join([
                    line.strip().replace("➡️", "").strip()
                    for line in reasoning_log.strip().splitlines()
                    if line.strip()
                ])
                reasoning_msg = f"🧠 `{label}` Agent → {steps_inline}"
                st.chat_message("ai").write(reasoning_msg)
                st.session_state.messages.append({"role": "ai", "content": reasoning_msg})

            # Final response
            st.chat_message("ai").write(response)
            st.session_state.messages.append({"role": "ai", "content": response})

        except Exception as e:
            st.error(f"⚠️ An error occurred while handling the query: {e}")

# Sidebar download
if "latest_file_content" in st.session_state:
    with st.sidebar.expander(f"💾 {st.session_state.latest_file_label}", expanded=False):
        st.download_button(
            label=st.session_state.latest_file_label,
            data=st.session_state.latest_file_content,
            file_name=st.session_state.latest_file_name,
            mime="text/plain"
        )