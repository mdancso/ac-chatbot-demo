from typing import TypedDict, List, Union, AsyncGenerator

from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentFinish
from langchain_core.messages import BaseMessage
from langchain.schema import Document as LangChainDocument
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult
from models.util.retriever_with_self_reflection import get_retriever_with_self_reflection
from models.util.archicad_agent import get_archicad_tools_agent
from models.util.generator_with_self_reflection import get_generator_with_self_reflection

class SelfReflectAgentRAG(RAGChatModel):
    class GraphState(TypedDict):
        # Original question
        question : str
        # Generated answer that may change during traversal
        answer: Union[str, None]
        # The list of previous messages in the conversation
        chat_history: list[BaseMessage]
        # List of past queries (the last element is always the current query)
        query_history: List[str]
        # Retrieved documents
        documents : List[LangChainDocument]

    def __init__(self, retriever, model="gpt-4o-mini", secondary_model="gpt-4o-mini", **kwargs):
        # lego pieces
        front_end_agent = get_archicad_tools_agent(model)
        retriever_with_self_reflection = get_retriever_with_self_reflection(retriever, secondary_model)
        generation_with_self_reflection = get_generator_with_self_reflection(model, secondary_model)

        # Main Graph
        
        # Setup node
        def setup_node(state):
            question = state.get("question")
            assert question is not None

            chat_history = state.get("chat_history")

            return {
                "question": question,
                "chat_history": chat_history if chat_history is not None else [],
                "query_history": [],
                "documents": [],
            }

        # Agent node
        def parse_agent_output(agent_output):
            if isinstance(agent_output, AgentFinish):
                return {"answer": agent_output.return_values["output"]}
            else:
                # Can only be a single ToolCall inside a list
                return {"query_history": [agent_output[0].tool_input["query"]]}
                
        self.agent_node = RunnablePassthrough.assign(intermediate_steps=RunnableLambda(lambda _state: [])) \
            | front_end_agent | parse_agent_output

        def error_handling_node(state):
            if not state["documents"]:
                return {"answer": "Couldn't find relevant information in the database!"}
            elif not state["answer"]:
                return {"answer": "Couldn't generate answer from given context!"}

        # Define the new graph
        workflow = StateGraph(SelfReflectAgentRAG.GraphState)

        # Add Nodes
        self.INIT_NODE = "init"
        self.AGENT_NODE = "agent"
        self.RETRIEVER_NODE = "retriever"
        self.GENERATION_NODE = "generate"
        self.ERROR_HANDLING_NODE = "error_handling"

        workflow.add_node(self.INIT_NODE, setup_node)
        workflow.add_node(self.AGENT_NODE, self.agent_node)
        workflow.add_node(self.RETRIEVER_NODE, retriever_with_self_reflection)
        workflow.add_node(self.GENERATION_NODE, generation_with_self_reflection)
        workflow.add_node(self.ERROR_HANDLING_NODE, error_handling_node)

        # Add connections
        workflow.set_entry_point(self.INIT_NODE)
        workflow.add_edge(self.INIT_NODE, self.AGENT_NODE)
        is_direct_answer = lambda state: "yes" if isinstance(state.get("answer"), str) else "no"
        workflow.add_conditional_edges(
            self.AGENT_NODE,
            is_direct_answer,
            {
                "yes": END,
                "no": self.RETRIEVER_NODE,
            },
        )
        workflow.add_conditional_edges(
            self.RETRIEVER_NODE,
            lambda state: "found_relevant_info" if len(state["documents"]) > 0 else "no_relevant_info",
            {
                "found_relevant_info": self.GENERATION_NODE,
                "no_relevant_info": self.ERROR_HANDLING_NODE,
            },
        )
        workflow.add_conditional_edges(
            self.GENERATION_NODE,
            lambda state: "valid_answer" if state["answer"] else "invalid_answer",
            {
                "valid_answer": END,
                "invalid_answer": self.ERROR_HANDLING_NODE,
            },
        )
        workflow.add_edge(self.ERROR_HANDLING_NODE, END)

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
            if self.GENERATION_NODE in chunk:
                answer = chunk[self.GENERATION_NODE]["answer"]
            elif self.AGENT_NODE in chunk:
                agent_answer = chunk[self.AGENT_NODE].get("answer")
                # Agent can reply with direct answer or ToolCall
                if agent_answer:
                    answer = agent_answer
            elif self.ERROR_HANDLING_NODE in chunk:
                answer = chunk[self.ERROR_HANDLING_NODE]["answer"]

            if answer:
                for char in answer:
                    yield LLMAnswer(answer=char) 
