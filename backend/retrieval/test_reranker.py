
from services.embeddings import generate_embeddings

from retrieval.vector_store import VectorStore
from retrieval.bm25_search import BM25Search
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import MedCPTReranker


# STEP 1: Sample research documents

chunks = [
    "Artificial intelligence is used in healthcare.",
    "Machine learning helps computers learn from data.",
    "RAG combines information retrieval with language generation.",
    "Deep learning is a subset of machine learning.",
    "Healthcare systems use artificial intelligence for diagnosis.",
    "Natural language processing helps computers understand text."
]


# STEP 2: Generate document embeddings

print("Generating document embeddings...")

embeddings = generate_embeddings(chunks)

print("Document embeddings generated successfully!")


# STEP 3: Create FAISS vector store

print("Creating FAISS vector store...")

vector_store = VectorStore()

vector_store.add_documents(chunks, embeddings)

print("FAISS indexing completed!")


# STEP 4: Create BM25 search

print("Creating BM25 search...")

bm25_search = BM25Search()

bm25_search.add_documents(chunks)

print("BM25 indexing completed!")


# STEP 5: Initialize Hybrid Search

hybrid_search = HybridSearch(
    vector_store,
    bm25_search
)

print("Hybrid Search initialized successfully!")


# STEP 6: User query

query = "How is artificial intelligence used in healthcare?"


# STEP 7: Generate query embedding

print("\nGenerating query embedding...")

query_embedding = generate_embeddings([query])[0]


# STEP 8: Retrieve documents using Hybrid Search

print("\nPerforming Hybrid Search...")

hybrid_results = hybrid_search.search(
    query=query,
    query_embedding=query_embedding,
    top_k=5
)


# STEP 9: Display Hybrid Search results

print("\n========== HYBRID SEARCH RESULTS ==========")

if not hybrid_results:
    print("No relevant documents found.")

else:

    for rank, result in enumerate(hybrid_results, start=1):

        print(f"\nRank: {rank}")
        print("Chunk:", result["chunk"])
        print("RRF Score:", result["rrf_score"])


# STEP 10: Initialize MedCPT Reranker

print("\nLoading MedCPT Cross-Encoder...")

reranker = MedCPTReranker()


# STEP 11: Rerank retrieved documents

print("\nReranking retrieved documents...")

reranked_results = reranker.rerank(
    query=query,
    documents=hybrid_results,
    top_k=5
)


# STEP 12: Display reranked results

print("\n========== MEDCPT RERANKED RESULTS ==========")

if not reranked_results:
    print("No relevant documents available for reranking.")

else:

    for rank, result in enumerate(reranked_results, start=1):

        print(f"\nRank: {rank}")
        print("Chunk:", result["chunk"])
        print("RRF Score:", result["rrf_score"])
        print("Rerank Score:", result["rerank_score"])

print("\n========== RERANKING COMPLETED ==========")
