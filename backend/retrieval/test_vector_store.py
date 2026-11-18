
from services.embeddings import generate_embeddings
from retrieval.vector_store import VectorStore


# STEP 1: Sample research documents

chunks = [
    "Artificial intelligence is used in healthcare.",
    "Machine learning helps computers learn from data.",
    "RAG combines information retrieval with language generation."
]


# STEP 2: Generate document embeddings

print("Generating document embeddings...")

embeddings = generate_embeddings(chunks)

print("Document embeddings generated successfully!")


# STEP 3: Create FAISS vector store

print("\nCreating FAISS vector store...")

vector_store = VectorStore()

# Add documents and embeddings
vector_store.add_documents(chunks, embeddings)

print("Documents indexed successfully!")


# STEP 4: Define search question

query = "How is artificial intelligence used?"


# STEP 5: Generate query embedding

print("\nGenerating query embedding...")

query_embedding = generate_embeddings([query])[0]


# STEP 6: Search using FAISS

print("\nSearching FAISS vector store...")

results = vector_store.search(
    query_embedding,
    top_k=2
)


# STEP 7: Display search results

print("\n========== FAISS SEARCH RESULTS ==========")

if not results:

    print("No relevant documents found.")

else:

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("Chunk:", result["chunk"])
        print("Similarity Score:", result["score"])

print("\n========== FAISS SEARCH COMPLETED ==========")
