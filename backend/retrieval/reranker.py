import numpy as np
import torch
from sentence_transformers import CrossEncoder


class MedCPTReranker:
    def __init__(self, model_name="ncbi/MedCPT-Cross-Encoder"):
        """
        Initialize the MedCPT Cross-Encoder model.
        """
        print("[RERANKER] Loading MedCPT model...", flush=True)

        self.model = CrossEncoder(
            model_name,
            max_length=512
        )

        print("[RERANKER] MedCPT model loaded successfully.", flush=True)

    def _get_text(self, document):
        """
        Extract passage text from a retrieved document.
        """
        if not isinstance(document, dict):
            return ""

        metadata = document.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        text = (
            document.get("chunk")
            or document.get("text")
            or document.get("chunk_text")
            or metadata.get("chunk")
            or metadata.get("text")
            or metadata.get("chunk_text")
            or ""
        )

        return str(text).strip()

    def _safe_float(self, value, default=0.0):
        """
        Convert a value into a safe float.
        """
        try:
            value = float(value)

            if np.isfinite(value):
                return value

            return default

        except (TypeError, ValueError):
            return default

    def _sigmoid(self, score):
        """
        Convert a raw score into a value between 0 and 1.

        Note: This is only a score transformation.
        It is not a calibrated probability.
        """
        score = self._safe_float(score)

        score = np.clip(score, -50, 50)

        return float(1 / (1 + np.exp(-score)))

    def rerank(self, query, documents, top_k=5):
        """
        Rerank retrieved passages using MedCPT.

        Parameters:
            query: User's research question.
            documents: Passages returned by hybrid search.
            top_k: Number of final passages to return.

        Returns:
            Documents sorted by MedCPT relevance score.
        """

        print("\n[RERANKER] Starting reranking...", flush=True)

        if not query or not isinstance(query, str):
            print("[RERANKER] Invalid query.", flush=True)
            return []

        if not documents:
            print("[RERANKER] No documents received.", flush=True)
            return []

        if not isinstance(documents, list):
            print("[RERANKER] Documents must be a list.", flush=True)
            return []

        top_k = max(1, int(top_k))

        valid_documents = []
        sentence_pairs = []

        for document in documents:

            if not isinstance(document, dict):
                continue

            passage = self._get_text(document)

            if not passage:
                continue

            valid_documents.append(document)
            sentence_pairs.append((query, passage))

        if not valid_documents:
            print(
                "[RERANKER] No valid passage text found.",
                flush=True
            )
            return []

        print(
            f"[RERANKER] Reranking {len(valid_documents)} passages...",
            flush=True
        )

        try:
            raw_scores = self.model.predict(
                sentence_pairs,
                activation_fct=torch.nn.Identity(),
                show_progress_bar=False
            )

            raw_scores = np.asarray(raw_scores).reshape(-1)

        except Exception as error:
            print(
                f"[RERANKER ERROR] Model prediction failed: {error}",
                flush=True
            )
            raise

        if len(raw_scores) != len(valid_documents):
            raise ValueError(
                "Number of reranker scores does not match "
                "the number of documents."
            )

        reranked_documents = []

        for document, raw_score in zip(valid_documents, raw_scores):

            raw_score = self._safe_float(raw_score)

            # Create a copy so the original search result is not modified.
            updated_document = document.copy()

            # Preserve existing metadata.
            metadata = updated_document.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            updated_document["metadata"] = metadata.copy()

            # Store reranker scores.
            updated_document["rerank_logit"] = raw_score
            updated_document["medcpt_raw_score"] = raw_score
            updated_document["rerank_score"] = self._sigmoid(raw_score)
            updated_document["reranker_score"] = self._sigmoid(raw_score)

            reranked_documents.append(updated_document)

        # Sort by raw MedCPT score in descending order.
        reranked_documents.sort(
            key=lambda item: item["rerank_logit"],
            reverse=True
        )

        final_results = reranked_documents[:top_k]

        print(
            f"[RERANKER] Reranking completed. "
            f"Returning top {len(final_results)} passages.",
            flush=True
        )

        for rank, document in enumerate(final_results, start=1):

            print(
                f"[RERANKER] Rank {rank} | "
                f"Raw score: {document['rerank_logit']:.4f} | "
                f"Display score: {document['rerank_score']:.4f}",
                flush=True
            )

        return final_results
