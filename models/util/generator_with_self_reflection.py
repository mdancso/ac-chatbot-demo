from typing import TypedDict, List, Union
from langgraph.graph import END, StateGraph
from langchain.schema import Document as LangChainDocument
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from models.util.answer_grader import get_answer_grader
from models.util.answer_generator import get_answer_generator
from models.util.hallucination_grader import get_hallucination_grader

def get_generator_with_self_reflection(model, secondary_model):
    generation_chain = get_answer_generator(model)
    hallucination_grader = get_hallucination_grader(secondary_model)
    answer_grader = get_answer_grader(secondary_model)

    class GenerationGraphState(TypedDict):
        # original question
        question: str

        # Retrieved documents
        documents : List[LangChainDocument]

        # Generated answer that may change during traversal
        answer: Union[str, None]

        # Number of generations
        num_gen: int

    # init
    def init_state(state):
        return {"num_gen": 0}

    # generation
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    generation_node = \
        RunnablePassthrough.assign(
            answer = \
                RunnableLambda(lambda state: {"question": state["question"], "context": format_docs(state["documents"])}) \
                | generation_chain
        ) | RunnableLambda(lambda state: {"answer": state["answer"], "num_gen": state["num_gen"] + 1})

    # hallucination grader
    is_hallucination = \
        RunnableLambda(
            lambda state: {
                "documents": format_docs(state["documents"]),
                "generation": state["answer"]
            }
        ) \
        | hallucination_grader \
        | RunnableLambda(lambda grade_res: "yes" if grade_res.binary_score == "no" else "no") # yes | no (has to be inverted, because yes means answ is grounded.)

    # answer grader
    answers_question = \
        RunnableLambda(
            lambda state: {
                "question": state["question"],
                "generation": state["answer"]
            }
        ) \
        | answer_grader \
        | RunnableLambda(lambda grade_res: grade_res.binary_score)

    # Safeguard to avoid infinite loop
    # if num_gen is above threshold remove answer
    def safeguard_node(state):
        new_answer = state["answer"]
        if state["num_gen"] >= 2:
            new_answer = None
        return {"answer": new_answer}

    # Define a new graph
    workflow = StateGraph(GenerationGraphState)

    GENERATE_NODE = "generate"
    DUMMY_NODE = "dummy"
    INIT_NODE = "init"
    SAFEGUARD_NODE = "safeguard"

    workflow.add_node(INIT_NODE, init_state)
    workflow.add_node(GENERATE_NODE, generation_node)
    workflow.add_node(DUMMY_NODE, lambda state: state)
    workflow.add_node(SAFEGUARD_NODE, safeguard_node)

    workflow.set_entry_point(INIT_NODE)
    workflow.add_edge(INIT_NODE, GENERATE_NODE)
    workflow.add_conditional_edges(
        GENERATE_NODE,
        is_hallucination,
        {
            "yes": SAFEGUARD_NODE,
            "no": DUMMY_NODE,
        },
    )
    workflow.add_conditional_edges(
        DUMMY_NODE,
        answers_question,
        {
            "yes": END,
            "no": SAFEGUARD_NODE,
        },
    )
    workflow.add_conditional_edges(
        SAFEGUARD_NODE,
        lambda state: "can_continue" if state["answer"] else "too_many_gen_attempts",
        {
            "can_continue": GENERATE_NODE,
            "too_many_gen_attempts": END,
        },
    )

    return workflow.compile()