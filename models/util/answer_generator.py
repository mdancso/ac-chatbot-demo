from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.util.prompts import Prompts
from models.util.llms import ChatModels, PossibleModels

def get_answer_generator(model: PossibleModels):
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
    generation_llm = ChatModels.get(model)

    # Chain
    return generation_prompt | generation_llm | StrOutputParser()
