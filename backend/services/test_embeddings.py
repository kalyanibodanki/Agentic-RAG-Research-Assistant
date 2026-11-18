
from embeddings import generate_embeddings


sample_chunks = [
    "Agentic RAG uses an intelligent agent to retrieve information.",
    "Retrieval Augmented Generation combines retrieval and generation.",
    "Machine learning allows computers to learn from data."
]


embeddings = generate_embeddings(sample_chunks)


print("\nEmbedding generation successful!")

print("Number of chunks:", len(sample_chunks))

print("Embedding shape:", embeddings.shape)

print("\nFirst chunk embedding:")
print(embeddings[0])

print("\nEmbedding dimension:", len(embeddings[0]))
