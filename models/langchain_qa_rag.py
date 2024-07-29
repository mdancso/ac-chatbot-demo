from langchain_openai import ChatOpenAI
from typing import AsyncGenerator, Union
from langchain.prompts import SystemMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain as ConvRetrievalChain

from models.util.prompts import Prompts
from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult

class LangChainDocumentQaRAG(RAGChatModel):
    def __init__(self, retriever, model: str="gpt-4o-mini"):
        qa = ConvRetrievalChain.from_llm(
            ChatOpenAI(model_name=model, streaming=True, temperature = 0),
            retriever=retriever,
            memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer', k=0),
            return_source_documents=True,
            chain_type="stuff"
        )
        system_message = Prompts.ARCHICAD_RETRIEVER_CHATBOT_SYSTEM_PROMPT
        
        qa.combine_docs_chain.llm_chain.prompt.messages.insert(0,  SystemMessagePromptTemplate.from_template(system_message))
        self.chain = qa
    
    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        inputs = {"question": question, "chat_history": memory.get_langchain_messages()}
        res = self.chain.invoke(inputs)
        return RAGResult(
            question=question,
            answer=res["answer"],
            context=[doc.page_content for doc in res["source_documents"]]
        )

    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        raise NotImplementedError("TODO")
    

