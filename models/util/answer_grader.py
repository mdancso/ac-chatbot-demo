from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

def get_answer_grader(model):
    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    # LLM with function call 
    answer_grader_llm = ChatOpenAI(model=model, temperature=0)
    structured_answer_grader_llm = answer_grader_llm.with_structured_output(GradeAnswer)

    # Prompt 
    answer_grader_system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_grader_system_prompt),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_grader_prompt | structured_answer_grader_llm
    