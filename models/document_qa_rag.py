from langchain_openai import ChatOpenAI
from typing import AsyncGenerator, Union
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models.util.prompts import Prompts
from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult

class DocumentQaRAG(RAGChatModel):
    def __init__(self, retriever, model="gpt-3.5-turbo"):
        def format_docs(docs):
            """Default way to format documents for prompt injection."""
            return "\n\n".join(doc.page_content for doc in docs)

        # Set model for all LLM calls
        llm = ChatOpenAI(model_name=model, temperature=0)

        # Contextualize Question
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain # if a Chain is returned, it will be executed
            else:
                return input["question"]

        # RAG QA Chain
        qa_system_prompt = Prompts.ARCHICAD_ANSWER_GEN_PROMPT + "\n\nContext:\n{context}"
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        custom_rag_qa_chain = (
            RunnablePassthrough.assign(
                contextual_question = contextualized_question
            ).assign(
                context = RunnableLambda(lambda x: x["contextual_question"]) | retriever,
            ).assign(
                answer = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | qa_prompt | llm | StrOutputParser()
            )
        )

        self.chain = custom_rag_qa_chain

    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        inputs = {"question": question, "chat_history": memory.get_langchain_messages()}
        res = self.chain.invoke(inputs)
        return RAGResult(
            question=question,
            answer=res["answer"],
            context=[doc.page_content for doc in res["context"]]
            )

    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        contextual_question = ""
        async for chunk in self.chain.astream({"question": question, "chat_history": memory.get_langchain_messages()}):
            ans_chunk = chunk.get("answer")
            if ans_chunk:
                yield LLMAnswer(answer=ans_chunk)

            contextual_question_chunk = chunk.get("contextual_question")
            if contextual_question_chunk:
                contextual_question += contextual_question_chunk
                continue
            
            context = chunk.get("context")
            if context:
                yield ToolCall(
                    name="retriever_tool",
                    query=contextual_question,
                    documents=[
                        Document(content=doc.page_content, metadata=doc.metadata)
                        for doc in context
                    ]
                )         
    