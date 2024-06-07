from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, create_openai_functions_agent
from langchain.tools import tool

from models.util.prompts import Prompts

def get_tool():
    # This tool is only a placeholder
    # If the Agent chooses to run this tool, we start executing a much more complex part of the graph
    class RetrieverQuery(BaseModel):
        query: str = Field(description="Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history.")

    @tool("multiplication-tool", args_schema=RetrieverQuery)
    def langchain_retriever_tool(query: str) -> list:
        """Searches and returns information about Archicad."""
        pass

    return langchain_retriever_tool

def get_prompt():
    # Loosly based on hwchase17/openai-functions-agent
    return ChatPromptTemplate.from_messages(
        [
            ("system", Prompts.ARCHICAD_RETRIEVER_CHATBOT_SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

def get_archicad_functions_agent(model):
    """Front-end archicad agent implementation, that can call a retriever tool. This version uses the Openai function calling mechanism."""
    agent_tools = [get_tool()]
    agent_llm = ChatOpenAI(model=model, streaming=True)
    return create_openai_functions_agent(agent_llm.with_config({"tags": ["agent_llm"]}), agent_tools, get_prompt())

def get_archicad_tools_agent(model):
    """Front-end archicad agent implementation, that can call a retriever tool. This version uses the Openai tool calling mechanism."""
    agent_tools = [get_tool()]
    agent_llm = ChatOpenAI(model=model, streaming=True)
    return create_openai_tools_agent(agent_llm.with_config({"tags": ["agent_llm"]}), agent_tools, get_prompt())
