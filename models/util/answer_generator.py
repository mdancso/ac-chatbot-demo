from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from models.util.prompts import Prompts

def get_answer_generator(model):
    # Prompt
    generation_system_prompt = Prompts.ARCHICAD_ANSWER_GEN_PROMPT + "\n\nContext:\n{context}"
    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generation_system_prompt),
            #MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # Alternatively you can use a default prompt that doesn't have specific instructions regarding role nor chat history
    # prompt = hub.pull("rlm/rag-prompt")

    # LLM
    generation_llm = ChatOpenAI(model=model, temperature=0)

    # Chain
    return generation_prompt | generation_llm | StrOutputParser()