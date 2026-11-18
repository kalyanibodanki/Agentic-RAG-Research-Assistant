
from retrieval.bm25_search import BM25Search

# Sample research documents
chunks = [
    "Artificial intelligence is used in healthcare.",
    "Machine learning helps computers learn from data.",
    "RAG combines information retrieval with language generation.",
    "Deep learning is a subset of machine learning.",
    "Healthcare systems use artificial intelligence for diagnosis.",
    "Natural language processing helps computers understand text."
]

# Create BM25 search
bm25_search = BM25Search()

# Add documents
bm25_search.add_documents(chunks)

# Search query
query = "healthcare"

results = bm25_search.search(query, top_k=3)

print("\nBM25 Search Results:")

for result in results:
    print("\nChunk:", result["chunk"])
    print("BM25 Score:", result["score"])
