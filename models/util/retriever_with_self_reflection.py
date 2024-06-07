
import operator
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated, List
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document as LangChainDocument

from models.util.retrieval_grader import get_retriever_grader
from models.util.question_rewriter import get_question_rewriter

def get_retriever_with_self_reflection(retriever, model):
    retrieval_grader = get_retriever_grader(model)
    question_rewriter = get_question_rewriter(model)

    class RetrieverGraphState(TypedDict):
        # original question
        question: str

        # List of past queries (the last element is always the current query)
        query_history: Annotated[List[str], operator.add]

        # Retrieved documents (Can be empty if no relevant documents found)
        documents : List[LangChainDocument]

    # db
    database_node = \
        RunnableLambda(lambda state: state["query_history"][-1]) \
        | retriever \
        | RunnableLambda(lambda docs: {"documents": docs})

    # self reflection
    are_documents_relevant = \
        RunnableLambda(
            lambda state: {
                "question": state["question"],
                "document": "\n".join([doc.page_content for doc in state["documents"]])
                }) \
        | retrieval_grader \
        | RunnableLambda(lambda grade_res: "yes" if grade_res.relevant else "no")

    # rewrite question
    new_query_gen = RunnableLambda(
        lambda state: {"question": state["query_history"][-1]}
        ) | question_rewriter

    query_rewriter_node = new_query_gen | RunnableLambda(lambda q: {"query_history": [q]})
        
    # check stop criteria
    def safeguard_node(state):
        new_docs = state["documents"]
        if len(state["query_history"]) >= 2:
            new_docs = []
        return {"documents": new_docs}

    # Define a new graph
    workflow = StateGraph(RetrieverGraphState)

    DB_NODE = "db"
    QUERY_REWRITE_NODE = "query_rewrite"
    SAFEGUARD_NODE = "safeguard"

    workflow.add_node(DB_NODE, database_node)
    workflow.add_node(QUERY_REWRITE_NODE, query_rewriter_node)
    workflow.add_node(SAFEGUARD_NODE, safeguard_node)

    workflow.set_entry_point(DB_NODE)
    workflow.add_conditional_edges(
        DB_NODE,
        are_documents_relevant,
        {
            "yes": END,
            "no": QUERY_REWRITE_NODE,
        },
    )
    workflow.add_edge(QUERY_REWRITE_NODE, SAFEGUARD_NODE)
    workflow.add_conditional_edges(
        SAFEGUARD_NODE,
        lambda state: "no_documents" if len(state["documents"]) == 0 else "continue",
        {
            "no_documents": END,
            "continue": DB_NODE,
        },
    )

    return workflow.compile()