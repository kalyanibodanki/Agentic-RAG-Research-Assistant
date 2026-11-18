
from sentence_transformers import SentenceTransformer


# ==============================================
# MODEL CONFIGURATION
# ==============================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384

print("\n========== LOADING EMBEDDING MODEL ==========")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")

print("Model:", MODEL_NAME)

print("Embedding dimension:", EMBEDDING_DIMENSION)

print("=============================================\n")


# ==============================================
# GENERATE EMBEDDINGS
# ==============================================

def generate_embeddings(chunks):

    """
    Generate normalized embeddings for research
    document chunks or user questions.

    Returns:
        NumPy array with shape (number_of_chunks, 384)
    """

    if not chunks:
        return []

    if not isinstance(chunks, list):
        raise ValueError(
            "Input must be a list of text strings."
        )

    if not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError(
            "Every input item must be a string."
        )

    if not all(chunk.strip() for chunk in chunks):
        raise ValueError(
            "Input contains an empty text string."
        )

    embeddings = model.encode(

        chunks,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=True,

        batch_size=32

    )

    return embeddings
