import operator
import itertools
from typing import TypedDict, Annotated, Union, AsyncGenerator

from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor

from models.util.prompts import Prompts
from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult

class AgenticRAG(RAGChatModel):
    name = "Smart Query Agent"
    info = "This variant uses an intelligent AI agent to dynamically decide when to retrieve information and autonomously generate tailored queries, enhancing adaptability and efficiency in conversations."

    class AgentState(TypedDict):
        # The input string
        input: str
        # The list of previous messages in the conversation
        chat_history: list[BaseMessage]
        # The outcome of a given call to the agent
        # Needs `None` as a valid type, since this is what this will start as
        agent_outcome: Union[AgentAction, AgentFinish, None]
        # List of actions and corresponding observations
        # Here we annotate this with `operator.add` to indicate that operations to
        # this state should be ADDED to the existing values (not overwrite it)
        intermediate_steps: Annotated[list[tuple[AgentAction, list]], operator.add]

    def __init__(self, retriever, model="gpt-4o-mini"):
        # Define tools node
        @tool
        def archicad_retriever_tool(query: str) -> list:
            """Searches and returns information about Archicad."""
            docs = retriever.invoke(query)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        tools = [archicad_retriever_tool]
        tool_executor = ToolExecutor(tools)

        def execute_tools(data):
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            intermediate_steps = []
            for action in data["agent_outcome"]:
                tool_call = ToolInvocation(tool=action.tool, tool_input=action.tool_input)
                output = tool_executor.invoke(tool_call)
                intermediate_steps.append((action, output))

            return {"intermediate_steps": intermediate_steps}

        # Define Agent node
        # non archicad specific version: prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Prompts.ARCHICAD_RETRIEVER_CHATBOT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = ChatOpenAI(model=model, streaming=True)
        agent = create_openai_tools_agent(llm.with_config({"tags": ["agent_llm"]}), tools, prompt)

        run_agent = agent | RunnableLambda(lambda res: {"agent_outcome": res})

        # Define logic that will be used to determine which conditional edge to go down
        def should_continue(data):
            # If the agent outcome is an AgentFinish, then we return `exit` string
            # This will be used when setting up the graph to define the flow
            if isinstance(data["agent_outcome"], AgentFinish) or len(data["intermediate_steps"]) > 2:
                return "end"
            # Otherwise, an AgentAction is returned
            # Here we return `continue` string
            # This will be used when setting up the graph to define the flow
            else:
                return "continue"

        # Define a new graph
        workflow = StateGraph(AgenticRAG.AgentState)
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        self.app = workflow.compile()

    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        inputs = {"input": question, "chat_history": memory.get_langchain_messages()}
        res = self.app.invoke(inputs)
        return RAGResult(
            question=question,
            answer=res["agent_outcome"].return_values["output"],
            context=list(itertools.chain.from_iterable(map(lambda x: x[1], res["intermediate_steps"])))
            )

    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        inputs = {"input": question, "chat_history": memory.get_langchain_messages()}
        async for output in self.app.astream_log(inputs, include_types=["llm"]):
            # astream_log() yields the requested logs (here LLMs) in JSONPatch format
            for op in output.ops:
                if op["path"] == "/streamed_output/-":
                    action = op["value"].get("action")
                    if action:
                        for tool_call_action, res in action["intermediate_steps"]:
                            yield ToolCall(
                                name=tool_call_action.tool,
                                query=tool_call_action.tool_input["query"],
                                documents=[Document(content=doc.get("content"), metadata=doc.get("metadata")) for doc in res]
                                )
                elif op["path"].startswith("/logs/") and op["path"].endswith(
                    "/streamed_output/-"
                ):
                    # because we chose to only include LLMs, these are LLM tokens
                    chunk = op["value"].content
                    if chunk: # exclude empty content response chunks (tool calls)
                        yield LLMAnswer(answer=chunk)
