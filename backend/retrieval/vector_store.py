
import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=384):

        self.dimension = dimension

        # Inner product index.
        # Normalized vectors make this equivalent
        # to cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        self.chunks = []
        self.metadata = []

    # ==============================================
    # ADD DOCUMENTS
    # ==============================================

    def add_documents(
        self,
        chunks,
        embeddings,
        metadatas=None
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return

        if metadatas is None:
            metadatas = [
                {} for _ in chunks
            ]

        if len(chunks) != len(metadatas):
            raise ValueError(
                "Number of chunks and metadata entries must match."
            )

        if not all(isinstance(item, str) for item in chunks):
            raise ValueError(
                "Every chunk must be a string."
            )

        if not all(isinstance(item, dict) for item in metadatas):
            raise ValueError(
                "Every metadata entry must be a dictionary."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"but received {embeddings.shape[1]}."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain invalid numeric values."
            )

        # Normalize document vectors for cosine similarity
        embeddings = embeddings.copy()

        faiss.normalize_L2(embeddings)

        # Add vectors to FAISS
        self.index.add(embeddings)

        self.chunks.extend(chunks)
        self.metadata.extend(metadatas)

        print(
            f"Added {len(chunks)} chunks to vector store."
        )

        print(
            "Total vectors:",
            self.index.ntotal
        )

    # ==============================================
    # VECTOR SEARCH
    # ==============================================

    def search(
        self,
        query_embedding,
        top_k=5,
        selected_document_ids=None
    ):

        if self.index.ntotal == 0 or top_k <= 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must be a one- or two-dimensional array."
            )

        if query_embedding.shape[0] != 1:
            raise ValueError(
                "Vector search expects exactly one query embedding."
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"but received {query_embedding.shape[1]}."
            )

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError(
                "Query embedding contains invalid numeric values."
            )

        # Normalize query vector
        query_embedding = query_embedding.copy()

        faiss.normalize_L2(query_embedding)

        selected_ids = None

        if selected_document_ids is not None:

            selected_ids = {
                str(document_id)
                for document_id in selected_document_ids
            }

        # Search the complete index so document filtering
        # does not accidentally remove eligible results.
        scores, indices = self.index.search(
            query_embedding,
            self.index.ntotal
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk_metadata = self.metadata[index]

            if selected_ids is not None:

                document_id = chunk_metadata.get(
                    "document_id"
                )

                if str(document_id) not in selected_ids:
                    continue

            results.append({

                "chunk": self.chunks[index],

                "score": float(score),

                "metadata": chunk_metadata

            })

            if len(results) >= top_k:
                break

        return results
