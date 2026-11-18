
from rank_bm25 import BM25Okapi
import re


class BM25Search:

    def __init__(self):

        self.bm25 = None

        self.chunks = []

        self.metadata = []

        self.tokenized_chunks = []

    # ==============================================
    # TOKENIZATION
    # ==============================================

    def tokenize(self, text):

        if not isinstance(text, str):
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # ==============================================
    # ADD DOCUMENTS
    # ==============================================

    def add_documents(
        self,
        chunks,
        metadatas=None
    ):

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

        self.chunks.extend(chunks)

        self.metadata.extend(metadatas)

        # Tokenize all stored chunks
        self.tokenized_chunks = [
            self.tokenize(chunk)
            for chunk in self.chunks
        ]

        # Ignore empty token lists
        valid_documents = [
            tokens
            for tokens in self.tokenized_chunks
            if tokens
        ]

        if not valid_documents:

            self.bm25 = None

            return

        # Rebuild BM25 index using all stored chunks
        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

        print(
            f"Added {len(chunks)} chunks to BM25 search."
        )

        print(
            "Total BM25 documents:",
            len(self.chunks)
        )

    # ==============================================
    # BM25 SEARCH
    # ==============================================

    def search(
        self,
        query,
        top_k=5,
        selected_document_ids=None
    ):

        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = self.tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = scores.argsort()[::-1]

        selected_ids = None

        if selected_document_ids is not None:

            selected_ids = {
                str(document_id)
                for document_id in selected_document_ids
            }

        results = []

        for index in ranked_indices:

            score = float(scores[index])

            # Ignore zero or negative BM25 scores
            if score <= 0:
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

                "score": score,

                "metadata": chunk_metadata

            })

            if len(results) >= top_k:
                break

        return results
