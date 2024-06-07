from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

def get_question_rewriter(model):
    # LLM 
    re_write_llm = ChatOpenAI(model=model, temperature=0)

    # Prompt 
    re_write_system_prompt = """You a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning.
        Keep in mind that both the original question is related to Archicad, so should the generated question."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", re_write_system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    return re_write_prompt | re_write_llm | StrOutputParser()