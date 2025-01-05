from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from chatbot.llm_setup import llm
from chatbot.vector_store_setup import retriever
from chatbot.utils import reciprocal_rank_fusion, format_docs
from chatbot.retrievers import retrieve_with_context_overlap
from chatbot.prompts import (
    query_rewriting_prompt,
    stepback_query_prompt,
    multiquery_generation_prompt,
    answer_prompt,
    hyde_prompt,
    support_llm_prompt,
)
from langchain_core.runnables import RunnableLambda, RunnableSequence

query_rewriting_chain = query_rewriting_prompt | llm | StrOutputParser()

hyde_chain = hyde_prompt | llm | StrOutputParser()

stepback_query_chain = stepback_query_prompt | llm | StrOutputParser()

multiquery_generation_chain = (
    multiquery_generation_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

rag_chain = RunnableSequence(
    {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "regular_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": itemgetter("question") | retriever | format_docs,
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
        "multiquery_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": (
                    itemgetter("question")
                    | multiquery_generation_chain
                    | retriever.map()
                    | reciprocal_rank_fusion
                    | format_docs
                ),
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
        "hyde_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": (
                    itemgetter("question") | hyde_chain | retriever | format_docs
                ),
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
        "query_rewriting_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": (
                    itemgetter("question")
                    | query_rewriting_chain
                    | retriever
                    | format_docs
                ),
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
        "stepback_query_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": (
                    itemgetter("question")
                    | stepback_query_chain
                    | retriever
                    | format_docs
                ),
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
        "keyword_answer": RunnableSequence(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": (
                    RunnableLambda(lambda data: itemgetter("question")(data))
                    | retrieve_with_context_overlap
                ),
            }
            | support_llm_prompt
            | llm
            | StrOutputParser(),
        ),
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)
