
from services.embeddings import generate_embeddings
from retrieval.vector_store import VectorStore
from retrieval.bm25_search import BM25Search
from retrieval.hybrid_search import HybridSearch


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

print("Documents added to FAISS successfully!")


# STEP 4: Create BM25 search

print("Creating BM25 search...")

bm25_search = BM25Search()

bm25_search.add_documents(chunks)

print("Documents added to BM25 successfully!")


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


# STEP 8: Perform Hybrid Search

print("Performing Hybrid Search...")

results = hybrid_search.search(
    query=query,
    query_embedding=query_embedding,
    top_k=5
)


# STEP 9: Display results

print("\n========== HYBRID SEARCH RESULTS ==========")

if not results:
    print("No relevant documents found.")

else:
    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("Chunk:", result["chunk"])
        print("RRF Score:", result["rrf_score"])

print("\n========== SEARCH COMPLETED ==========")
