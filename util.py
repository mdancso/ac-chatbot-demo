import json
import streamlit as st
from typing import List
from streamlit_feedback import streamlit_feedback
from models.util.data_models import LLMAnswer, ToolCall

def display_tool_call(key: str, tool_call: dict, placeholder=None):
    if placeholder is None:
        placeholder = st

    with placeholder.chat_message("tool_call", avatar="🛠"):
        with st.expander(f"`{tool_call.get('name')}` called with query: `{tool_call.get('query')}`"):
            documents = tool_call.get("documents")
            if len(documents) == 0:
                st.write([])
                
            for i, doc in enumerate(documents):
                metadata = doc.get("metadata")
                relevance = metadata.get("relevant")
                
                known_relevance_indicator = "✔️ Relevant" if relevance else "❌ Not relevant"
                relevance_indicator = "⚠️ Relevance unknown" if relevance is None else known_relevance_indicator

                # Create a unique key for each button
                button_key = key + str(i)

                @st.experimental_dialog(" ")
                def popup_content(header, content):
                    st.header(header)
                    st.write(content)

                header = f"Unknown source - {relevance_indicator}"
                if metadata['source'].endswith('.pdf'):
                    header = f"📄 {metadata['id']} - Page {metadata['page']} - {relevance_indicator}"
                elif "http" in metadata['source']:
                    header = f"🌐 [{metadata['title']}]({metadata['source']}) - {relevance_indicator}"
                
                if st.button(header, key=button_key):
                    popup_content(header, doc.get("content"))
                    

def display_tool_calls(key: str, tool_calls: List[dict], placeholder=None):
    if placeholder is None:
        placeholder = st

    for tool_call in tool_calls:
        display_tool_call(key, tool_call, placeholder)

def display_ai_message(message: str, placeholder=None, on_feedback=None, key=None):
    if placeholder is None:
        placeholder = st

    with placeholder.chat_message("assistant"):
        st.markdown(message)

    feedback = streamlit_feedback(
        key=key,
        feedback_type="thumbs",
        optional_text_label="[Optional explanation]",
    )
    if feedback and on_feedback:
        on_feedback(feedback)
        

def display_user_message(id: str, message: str, on_delete=None, placeholder=None):
    if placeholder is None:
        placeholder = st

    with placeholder.chat_message("user"):
        st.markdown(message)

    cols = placeholder.columns(7)
    if cols[-1].button("Delete", key=id+"_delete_button") and on_delete:
        on_delete(id)

async def display_streaming_content(key: str, async_stream, on_finish):
    tool_calls_placeholder = st.container()

    stream_placeholder = None

    tool_calls = []
    answer = ""
    async for part in async_stream:
        if isinstance(part,LLMAnswer):
            answer += part.answer
            if not stream_placeholder:
                with st.chat_message("assistant"):
                    stream_placeholder = st.empty()
            stream_placeholder.markdown(answer)
        if isinstance(part, ToolCall):
            display_tool_call(key, json.loads(part.to_json()), tool_calls_placeholder)
            tool_calls.append(part)

    on_finish(LLMAnswer(answer), tool_calls)
    