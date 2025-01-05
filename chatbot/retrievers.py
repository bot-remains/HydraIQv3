from langchain_core.documents import Document
from typing import List
from langchain_community.retrievers import BM25Retriever
from operator import itemgetter
from chatbot.vector_store_setup import vector_store, retriever
from langchain.retrievers import EnsembleRetriever


def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get("index") == target_index:
            return doc
    return None


def retrieve_with_context_overlap(query) -> List[str]:
    pinecone_index = vector_store._index
    total_docs = pinecone_index.describe_index_stats()["total_vector_count"]
    print(total_docs)
    # Perform similarity search for all documents (if needed)
    all_docs = vector_store.similarity_search("", k=total_docs)

    keyword_retriever = BM25Retriever.from_documents(all_docs)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            retriever,
            keyword_retriever,
        ],
        weights=[0.8, 0.2],
    )

    docs = ensemble_retriever.invoke(query)
    return docs
