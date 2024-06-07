import asyncio
from typing import AsyncGenerator, Union

from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult

class EchoBot(RAGChatModel):
    def __init__(self, retriever, **kwargs):
        self.retriever = retriever

    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        return RAGResult(question=question, answer=question, context=["echo: question"])

    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        print("EchoBot to your service:)")
        yield ToolCall(name="example_tool", query="example_query", documents=[Document(content="echo: question", metadata={"source": "echo"})])
        for char in question:
            await asyncio.sleep(0.05)
            yield LLMAnswer(answer=char)
