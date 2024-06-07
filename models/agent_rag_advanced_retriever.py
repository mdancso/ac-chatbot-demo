import operator
from typing import TypedDict, Annotated, List, Union, AsyncGenerator

from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema import Document as LangChainDocument

from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult
from models.util.archicad_agent import get_archicad_functions_agent
from models.util.retriever_with_self_reflection import get_retriever_with_self_reflection

class AgentRAGWithSelfReflectRetrieval(RAGChatModel):
    class AgentState(TypedDict):
        # The input string
        question: str

        # The list of previous messages in the conversation
        chat_history: list[BaseMessage]

        # Final answer
        answer: Union[str, None]

        # The outcome of a given call to the agent
        # Needs `None` as a valid type, since this is what this will start as
        agent_outcome: Union[AgentAction, AgentFinish, None]

        # List of actions and corresponding observations
        # Here we annotate this with `operator.add` to indicate that operations to
        # this state should be ADDED to the existing values (not overwrite it)
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

        # List of past queries (the last element is always the current query)
        query_history: List[str]

        # Retrieved documents
        documents : List[LangChainDocument]

    def __init__(self, retriever, model="gpt-3.5-turbo", secondary_model="gpt-3.5-turbo", **kwargs):
        # Setup node
        def setup_node(state):
            question = state.get("question")
            assert question is not None

            chat_history = state.get("chat_history")

            return {
                "question": question,
                "answer": None,
                "chat_history": chat_history if chat_history is not None else [],
                "query_history": [],
                "documents": [],
            }

        # Agent node
        front_end_agent = get_archicad_functions_agent(model)
        def run_agent(state):
            agent_outcome = front_end_agent.invoke(state)
            return {"agent_outcome": agent_outcome}

        # Retriever node
        retriever_with_self_reflection = get_retriever_with_self_reflection(retriever, secondary_model)
        def run_retriever(state):
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            agent_action = state["agent_outcome"]

            inp = {
                "question": state["question"],
                "query_history": [agent_action.tool_input["query"]]
            }

            output = retriever_with_self_reflection.invoke(inp)
            docs = output["documents"]
            content = "\n\n".join([doc.page_content for doc in docs])
            return {
                "intermediate_steps": [(agent_action, content)],
                "documents": docs,
                "query_history": output["query_history"]
            }

        # Output parse node
        def set_answer(state):
            # This node can receive the final answer either from the agent
            # or in case no relevant documents found a predefined text
            agent_output = state["agent_outcome"]
            if isinstance(agent_output, AgentFinish):
                return {"answer": agent_output.return_values["output"]}
            elif len(state["documents"]) == 0:
                return {"answer": "Couldn't find relevant information in the database!"}
            else:
                raise RuntimeError("Invalid model state")

        # Building the Graph
        workflow = StateGraph(AgentRAGWithSelfReflectRetrieval.AgentState)

        # Nodes
        self.INIT_NODE = "init"
        self.AGENT_NODE = "agent"
        self.RETRIEVER_NODE = "retriever"
        self.ANSWER_NODE = "answer "
        workflow.add_node(self.INIT_NODE, setup_node)
        workflow.add_node(self.AGENT_NODE, run_agent)
        workflow.add_node(self.RETRIEVER_NODE, run_retriever)
        workflow.add_node(self.ANSWER_NODE, set_answer)

        # Connections
        workflow.set_entry_point(self.INIT_NODE)
        workflow.add_edge(self.INIT_NODE, self.AGENT_NODE)
        is_direct_answer = lambda state: "yes" if isinstance(state["agent_outcome"], AgentFinish) else "no"
        workflow.add_conditional_edges(
            self.AGENT_NODE,
            is_direct_answer,
            {
                "yes": self.ANSWER_NODE,
                "no": self.RETRIEVER_NODE,
            },
        )
        if_relevant_info_in_db = lambda state: "yes" if len(state["documents"]) > 0 else "no"
        workflow.add_conditional_edges(
            self.RETRIEVER_NODE,
            if_relevant_info_in_db,
            {
                "yes": self.AGENT_NODE,
                "no": self.ANSWER_NODE,
            },
        )
        workflow.add_edge(self.ANSWER_NODE, END)

        # Compile graph
        self.app = workflow.compile()
        
    def invoke(self, question: str, memory: ChatMemory) -> RAGResult:
        inputs = {"question": question, "chat_history": memory.get_langchain_messages()}
        res = self.app.invoke(inputs)
        return RAGResult(
            question=question,
            answer=res["answer"],
            context=[doc.page_content for doc in res["documents"]]
            )

    async def stream_async(self, question: str, memory: ChatMemory) -> AsyncGenerator[Union[ToolCall, LLMAnswer], None]:
        # This model only does virtual streaming, beacuse self-reflection can mark answer as invalid
        for chunk in self.app.stream({"question": question, "chat_history": memory.get_langchain_messages()}):
            # Handle Tool call
            if self.RETRIEVER_NODE in chunk:
                state_update = chunk[self.RETRIEVER_NODE]
                yield ToolCall(
                    name=self.RETRIEVER_NODE,
                    query=str(state_update["query_history"]),
                    documents=[
                        Document(
                            content=doc.page_content,
                            metadata=doc.metadata
                        ) for doc in state_update["documents"]])

            # Handle generated answer
            answer = None
            if self.ANSWER_NODE in chunk:
                answer = chunk[self.ANSWER_NODE]["answer"]

            if answer:
                for char in answer:
                    yield LLMAnswer(answer=char) 
