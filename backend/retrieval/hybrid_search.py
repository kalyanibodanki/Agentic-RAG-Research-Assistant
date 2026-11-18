
"""Hybrid retrieval using Vector Search + BM25 + Reciprocal Rank Fusion."""


class HybridSearch:

    def __init__(self, vector_store, bm25_search):
        self.vector_store = vector_store
        self.bm25_search = bm25_search

    def _normalize_result(self, item):
        """Normalize search results into a common format."""

        if not isinstance(item, dict):
            return None

        result = dict(item)

        metadata = result.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)

        # Move important fields into metadata if necessary.
        for key in (
            "document_id",
            "filename",
            "chunk_index",
            "page_number"
        ):
            if result.get(key) is not None:
                metadata.setdefault(key, result[key])

        chunk = (
            result.get("chunk")
            or result.get("text")
            or result.get("chunk_text")
            or ""
        )

        result["chunk"] = str(chunk)
        result["text"] = str(chunk)
        result["metadata"] = metadata

        return result

    def _get_chunk_identity(self, result):
        """Generate a unique identity for each document chunk."""

        metadata = result.get("metadata", {})

        document_id = (
            metadata.get("document_id")
            or metadata.get("filename")
            or "unknown"
        )

        chunk_index = metadata.get("chunk_index")

        if chunk_index is not None:
            return str(document_id), str(chunk_index)

        chunk_text = result.get("chunk", "")

        return str(document_id), str(chunk_text)

    def _filter_documents(self, results, selected_document_ids):
        """Keep only results belonging to selected documents."""

        if not selected_document_ids:
            return results

        selected_ids = {
            str(document_id)
            for document_id in selected_document_ids
        }

        filtered = []

        for result in results:
            metadata = result.get("metadata", {})

            document_id = metadata.get("document_id")

            if document_id is not None and str(document_id) in selected_ids:
                filtered.append(result)

        return filtered

    def reciprocal_rank_fusion(
        self,
        vector_results,
        bm25_results,
        k=60
    ):
        """Combine vector and BM25 rankings using RRF."""

        fused_results = {}

        search_lists = [
            vector_results,
            bm25_results
        ]

        for results in search_lists:

            for rank, raw_result in enumerate(results, start=1):

                result = self._normalize_result(raw_result)

                if result is None:
                    continue

                identity = self._get_chunk_identity(result)

                if identity not in fused_results:

                    fused_results[identity] = {
                        "result": result,
                        "rrf_score": 0.0
                    }

                fused_results[identity]["rrf_score"] += (
                    1.0 / (k + rank)
                )

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        final_results = []

        for item in ranked_results:

            result = dict(item["result"])

            result["rrf_score"] = float(
                item["rrf_score"]
            )

            final_results.append(result)

        return final_results

    def search(
        self,
        query,
        query_embedding,
        top_k=30,
        selected_document_ids=None
    ):
        """Run vector search, BM25 search, and RRF fusion."""

        print(
            f"\n[HybridSearch] Searching: {query}",
            flush=True
        )

        print(
            f"[HybridSearch] Selected documents: {selected_document_ids}",
            flush=True
        )

        # Step 1: Vector search
        # Pass selected document IDs directly to VectorStore.
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            selected_document_ids=selected_document_ids
        )

        # Step 2: BM25 search
        # Pass selected document IDs directly to BM25Search.
        bm25_results = self.bm25_search.search(
            query,
            top_k=top_k,
            selected_document_ids=selected_document_ids
        )

        vector_results = vector_results or []
        bm25_results = bm25_results or []

        # Step 3: Normalize results
        vector_results = [
            self._normalize_result(item)
            for item in vector_results
        ]

        bm25_results = [
            self._normalize_result(item)
            for item in bm25_results
        ]

        vector_results = [
            item for item in vector_results
            if item is not None
        ]

        bm25_results = [
            item for item in bm25_results
            if item is not None
        ]

        # Step 4: Apply an additional document filter.
        vector_results = self._filter_documents(
            vector_results,
            selected_document_ids
        )

        bm25_results = self._filter_documents(
            bm25_results,
            selected_document_ids
        )

        # Step 5: Reciprocal Rank Fusion
        hybrid_results = self.reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )

        # Step 6: Display retrieval information
        print(
            f"[HybridSearch] Vector results: {len(vector_results)}",
            flush=True
        )

        print(
            f"[HybridSearch] BM25 results: {len(bm25_results)}",
            flush=True
        )

        print(
            f"[HybridSearch] Fused results: {len(hybrid_results)}",
            flush=True
        )

        final_results = hybrid_results[:top_k]

        print(
            f"[HybridSearch] Final results returned: {len(final_results)}",
            flush=True
        )

        return final_results