import operator
from typing import TypedDict, Annotated, List, Union, Literal, AsyncGenerator
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda, RunnableParallel

from models.util.memory import ChatMemory
from models.model_base import RAGChatModel
from models.util.data_models import ToolCall, LLMAnswer, Document, RAGResult
from models.util.retrieval_grader import get_retriever_grader
from models.util.archicad_agent import get_archicad_functions_agent

class AgentWithFallback(RAGChatModel):
    name = "Selective Agent"
    info = "This chatbot model evaluates the relevance of each retrieved document individually, ensuring that only the most pertinent information is used to generate responses, improving accuracy and detail."

    class ModelState(TypedDict):
        # Original question
        question: str

        # Query string for the vector db
        query: Union[str, None]

        # Final Answer
        answer: str

        # Retrieved documents
        documents : List[LangChainDocument]

        # Relevant documents (can be an empty list)
        relevant_documents : List[LangChainDocument]

        # Chat history
        chat_history : List[BaseMessage]

        # The outcome of a given call to the agent
        agent_outcome: Union[AgentAction, AgentFinish]
        
        # List of Tool call actions with their results
        intermediate_steps: Annotated[list[tuple[AgentAction, list]], operator.add]

    def __init__(self, retriever, model="gpt-4o-mini", secondary_model="gpt-4o-mini", **kwargs):
        # lego pieces
        front_end_agent = get_archicad_functions_agent(model)
        retrieval_grader = get_retriever_grader(secondary_model)

        # DB
        def retriever_from_db(state):
            query = state["agent_outcome"].tool_input["query"]
            docs = retriever.invoke(query)
            return {"query": query, "documents": docs}

        # Grader
        def grade_documents(state):
            """Attach metadata to documents whether they are relevant to answering the given question."""

            question = state["question"]
            docs = state["documents"]

            # Grading documents in parallel
            def grade(doc: LangChainDocument) -> LangChainDocument:
                doc = LangChainDocument(page_content=doc.page_content, metadata=doc.metadata.copy())
                doc.metadata["relevant"] = retrieval_grader.invoke({
                    "question": question, "document": doc.page_content
                }).relevant
                return doc

            graded_documents = list(RunnableParallel({
                str(i) : RunnableLambda(lambda docs, i=i: grade(docs[i])) for i in range(len(docs))
            }).invoke(docs).values())

            return {"documents": graded_documents}
            
        # Agent
        def run_agent(state):
            """"
            Runs the front end agent that interacts with the user.
            In case it called a tool it injects the retrieved relevant documents into the context.
            If no relevant documents were found it falls back to giving all documents to the model.
            This "fallback" behaviour can potentially be upgraded in the future with e.g.: Web-search
            """
            prev_outcome = state.get("agent_outcome")
            if prev_outcome and not isinstance(prev_outcome, AgentFinish):
                relevant_docs = list(filter(lambda doc: doc.metadata["relevant"], state["documents"]))

                # fallback
                if len(relevant_docs) == 0:
                    relevant_docs = state["documents"]

                state["intermediate_steps"] = [(prev_outcome, relevant_docs)]

            agent_outcome = front_end_agent.invoke(state)
            temp = {"agent_outcome": agent_outcome}
            if isinstance(agent_outcome, AgentFinish):
                temp["answer"] = agent_outcome.return_values["output"]
            return temp

        def is_direct_answer(data) -> Literal["yes", "no"]:
            return "yes" if isinstance(data["agent_outcome"], AgentFinish) else "no"

        # Define a new graph
        workflow = StateGraph(AgentWithFallback.ModelState)

        # Nodes
        self.AGENT_NODE = "Agent"
        self.RETRIEVER_NODE = "DB"
        self.GRADER_NODE = "Grader"

        workflow.add_node(self.AGENT_NODE, run_agent)
        workflow.add_node(self.RETRIEVER_NODE, retriever_from_db)
        workflow.add_node(self.GRADER_NODE, grade_documents)

        workflow.set_entry_point(self.AGENT_NODE)
        workflow.add_conditional_edges(
            self.AGENT_NODE,
            is_direct_answer,
            {
                "yes": END,
                "no": self.RETRIEVER_NODE,
            },
        )
        workflow.add_edge(self.RETRIEVER_NODE, self.GRADER_NODE)
        workflow.add_edge(self.GRADER_NODE, self.AGENT_NODE)

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
        query = ""
        for chunk in self.app.stream({"question": question, "chat_history": memory.get_langchain_messages()}):
            if self.RETRIEVER_NODE in chunk:
                query = chunk[self.RETRIEVER_NODE]["query"]
            elif self.GRADER_NODE in chunk:
                state_update = chunk[self.GRADER_NODE]
                yield ToolCall(
                    name="Database",
                    query=query,
                    documents=[
                        Document(
                            content=doc.page_content,
                            metadata=doc.metadata,
                        ) for doc in state_update["documents"]])
            elif self.AGENT_NODE in chunk:
                answer = chunk[self.AGENT_NODE].get("answer")
                if answer:
                    for char in answer:
                        yield LLMAnswer(answer=char)
            