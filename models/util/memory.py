import json
import uuid
import streamlit as st
from typing import List, Generator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from models.util.data_models import LLMAnswer, ToolCall

@st.cache_resource
def cache_memory():
    return ChatMemory()

class ChatMemory:
    def __init__(self):
        """
        Example schema:
        [
            {
                "id": "guid"
                "question": "What is Archicad?",
                "answer": "It a Graphisoft product.",
                "tools": [
                    {
                        "name": "retriever_tool"
                        "query": "Archicad",
                        "documents": {
                            "content": "...",
                            "metadata": {
                                ...
                            }
                        }
                    }
                ]
            }
        ]
        """
        self._messages: List[dict] = []

    def add_qa_pair(self, question: str, llm_answer: LLMAnswer, tools: List[ToolCall]):
        assert question is not None
        self._messages.append({
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": llm_answer.answer,
            "tools": [json.loads(tool_use.to_json()) for tool_use in tools]
        })

    def delete(self, id: str):
        for i, msg in enumerate(self._messages):
            if msg["id"] == id:
                self._messages = self._messages[0:i]
                break

    def attach_metadata(self, id: str, metadata: dict):
        for msg in self._messages:
            if msg["id"] == id:
                if "metadata" in msg:
                    msg.update(metadata)
                else:
                    msg["metadata"] = metadata
                break

    def clear(self):
        self._messages = []

    def get_langchain_messages(self) -> List[BaseMessage]:
        temp = []
        for msg in self._messages:
            temp.append(HumanMessage(msg.get("question")))
            temp.append(AIMessage(msg.get("answer")))
        return temp

    def loop_messages(self) -> Generator[dict, None, None]:
        for msg in self._messages:
            yield msg
