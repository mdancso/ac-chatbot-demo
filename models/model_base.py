from abc import ABC, abstractmethod
from typing import AsyncGenerator, Union

from models.util.memory import ChatMemory
from models.util.data_models import RAGResult, LLMAnswer, ToolCall
 
class RAGChatModel(ABC):
    @abstractmethod
    def __init__(self, retriever, **kwargs):
        """Constructor for RAG model accepts a LangChain retriever."""
        raise NotImplementedError(f"{self.__init__.__name__} method not implemented")
 
    @abstractmethod
    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        """Returns answer for question based on the information retrieved form the database."""
        raise NotImplementedError(f"{self.invoke.__name__} method not implemented")

    @abstractmethod
    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        """Streams answer in an async manner."""
        raise NotImplementedError(f"{self.stream_async.__name__} method not implemented")
