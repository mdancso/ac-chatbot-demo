from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

def get_hallucination_grader(model):
    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    # LLM with function call 
    hallucination_grader_llm = ChatOpenAI(model=model, temperature=0)
    structured_hallucination_grader_llm = hallucination_grader_llm.with_structured_output(GradeHallucinations)

    # Prompt 
    hallucination_system_promot = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_system_promot),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    # yes = grounded in facts so no hallucination
    return hallucination_prompt | structured_hallucination_grader_llm
