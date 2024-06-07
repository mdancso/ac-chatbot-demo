import uuid
import asyncio
import streamlit as st

from models.util.memory import cache_memory
from models.document_qa_rag import DocumentQaRAG
from models.agent_rag_advanced_retriever import AgentRAGWithSelfReflectRetrieval
from models.agent_with_fallback import AgentWithFallback
from models.agentic_rag import AgenticRAG

from database import load_db
from util import display_tool_calls, display_ai_message, display_user_message, display_streaming_content

# Init code

# Both of these calls are cached
vector_db = load_db()
memory = cache_memory()

# Initialize session state if not already done
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.selected_model_name = None
    st.session_state.how_many_docs_to_retrieve = 8
    st.session_state.chat_model = None
    st.session_state.choose_llm = "gpt-3.5-turbo"

    models = [
        DocumentQaRAG,
        AgentWithFallback,
        AgenticRAG,
        AgentRAGWithSelfReflectRetrieval
    ]
    st.session_state.models = {bot.__name__: bot for bot in models}

# Sidebar

## Chatbot Settings
st.sidebar.title("Chatbot")

if st.sidebar.button('Clear history'):
    memory.clear()

selected_model_name = st.sidebar.selectbox(
    'Select a chatbot:',
    st.session_state.models.keys()
)

choosen_llm = st.sidebar.radio(
    "Select which ",
    key="choose_llm",
    options=["gpt-3.5-turbo", "gpt-4o"],
)


how_many_docs_to_retrieve = st.sidebar.slider("How many documents should I retrieve per question?", 1, 10, st.session_state.how_many_docs_to_retrieve)

# Check if any parameter has changed
def parameters_changed():
    return (st.session_state.selected_model_name != selected_model_name or
            st.session_state.how_many_docs_to_retrieve != how_many_docs_to_retrieve or
            st.session_state.choosen_llm != choosen_llm)

if parameters_changed():
    st.session_state.selected_model_name = selected_model_name
    st.session_state.how_many_docs_to_retrieve = how_many_docs_to_retrieve
    st.session_state.choosen_llm = choosen_llm
    st.session_state.chat_model = st.session_state.models[selected_model_name](
        vector_db.as_retriever(k=how_many_docs_to_retrieve),
        model=choosen_llm
    )

st.sidebar.divider()

## Database Settings
st.sidebar.title("Database")

### File Uploader
with st.sidebar.form("my-form", clear_on_submit=True):
    files = st.file_uploader("upload files", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    if st.form_submit_button("Upload"):
        if 'files' not in st.session_state:
            st.session_state.files = []
        with st.spinner(text='Uploading...'):
            for file in files:
                vector_db.add_pdf_to_db(file.name, file.getvalue())

### Uploaded files list
st.sidebar.markdown("Uploaded files")
with st.sidebar:
    with st.spinner(text='Loading...'):
        known_documents = vector_db.get_known_documents()
        for index, doc_id in enumerate(known_documents):
            emp = st.sidebar.empty()
            col1, col2 = emp.columns([1, 1])
            col1.markdown(doc_id)
            #if col2.button("Del", key=f"but{index}"):
            #    vector_db.delete_file_from_db(doc_id)
            #    st.rerun()

st.sidebar.divider()

## Debug Info

st.sidebar.title("Debug Info")

@st.experimental_dialog("Chat history")
def chat_history():
    st.write(st.session_state)
    st.write(memory._messages)

if st.sidebar.button("Show History"):
    chat_history()

# Main Content

st.title(selected_model_name)

## Message history
for msg in memory.loop_messages():
    id_key = msg.get("id")

    def delete_msg(id):
        memory.delete(id)
        st.rerun()
    
    display_user_message(
        id=id_key,
        message=msg.get("question"),
        on_delete=delete_msg
    )
    display_tool_calls(id_key, msg.get("tools"))
    display_ai_message(
        msg.get("answer"),
        key=id_key,
        on_feedback=lambda feedback, id_key=id_key: memory.attach_metadata(id_key, {"feedback": feedback})
    )

## Chat interface
if question := st.chat_input("Ask Chatbot..."):
    temp_key_1 = str(uuid.uuid4())
    display_user_message(temp_key_1, question)
    
    async_stream = st.session_state.chat_model.stream_async(question, memory)
    temp_key_2 = str(uuid.uuid4())
    asyncio.run(display_streaming_content(temp_key_2, async_stream, on_finish=lambda llm_answ, tools: memory.add_qa_pair(question, llm_answ, tools)))
    st.rerun()
