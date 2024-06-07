from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

def get_retriever_grader(model):
    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        relevant: bool = Field(description="Documents are relevant to the question, True or False")

    # LLM with function call 
    grader_llm = ChatOpenAI(model=model, temperature=0)
    structured_llm_grader = grader_llm.with_structured_output(GradeDocuments)

    # Prompt 
    grader_system_message = """You are a grader assessing relevance of a retrieved document to a user question.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score True or False score to indicate whether the document is relevant to the question."""
    grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_message),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grader_prompt | structured_llm_grader