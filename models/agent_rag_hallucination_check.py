from typing import TypedDict, List, Union, AsyncGenerator

from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentFinish
from langchain.schema import Document as LangChainDocument
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult
from models.util.archicad_agent import get_archicad_tools_agent
from models.util.generator_with_self_reflection import get_generator_with_self_reflection

class AgentRAGWithHallucinationCheck(RAGChatModel):
    class AgentState(TypedDict):
        # Original question
        question : str
        # Generated answer that may change during traversal
        answer: Union[str, None]
        # The list of previous messages in the conversation
        chat_history: list[BaseMessage]
        # Query for vector db
        query: Union[str, None]
        # Retrieved documents
        documents : List[LangChainDocument]

    def __init__(self, retriever, model="gpt-4o-mini", secondary_model="gpt-4o-mini", **kwargs):
        # Setup node
        def setup_node(state):
            question = state.get("question")
            assert question is not None

            chat_history = state.get("chat_history")

            return {
                "question": question,
                "answer": None, # This doesn't have an effects as langgraph doesn't include it
                "chat_history": chat_history if chat_history is not None else [],
                "query": None,
                "documents": [],
            }

        # Agent node
        front_end_agent = get_archicad_tools_agent(model)

        def parse_agent_output(agent_output):
            if isinstance(agent_output, AgentFinish):
                return {"answer": agent_output.return_values["output"]}
            else:
                return {"query": agent_output[0].tool_input["query"]}
                
        agent_node = RunnablePassthrough.assign(intermediate_steps=RunnableLambda(lambda _state: [])) \
            | front_end_agent | parse_agent_output

        # Retrieval node
        def run_retriever(state):
            query = state["query"]
            assert query is not None
            return {"documents": retriever.invoke(query), "query": query}

        # Generation node
        generation_with_self_reflection = get_generator_with_self_reflection(model, secondary_model)

        def run_generation(state):
            answ = generation_with_self_reflection.invoke(state).get("answer")
            if not answ:
                answ = "Couldn't generate proper answer!"
            return {"answer": answ}

        # Define the graph
        workflow = StateGraph(AgentRAGWithHallucinationCheck.AgentState)

        # Add Nodes
        self.INIT_NODE = "init"
        self.AGENT_NODE = "agent"
        self.RETRIEVER_NODE = "retriever"
        self.GENERATION_NODE = "generate"
        self.ANSWER_NODE = "answer"

        workflow.add_node(self.INIT_NODE, setup_node)
        workflow.add_node(self.AGENT_NODE, agent_node)
        workflow.add_node(self.RETRIEVER_NODE, run_retriever)
        workflow.add_node(self.GENERATION_NODE, run_generation)

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
        workflow.add_edge(self.RETRIEVER_NODE, self.GENERATION_NODE)
        workflow.add_edge(self.GENERATION_NODE, END)

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
                    query=str(state_update.get("query")),
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

            if answer:
                for char in answer:
                    yield LLMAnswer(answer=char)  


            
            

