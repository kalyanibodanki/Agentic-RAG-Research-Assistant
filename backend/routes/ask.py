
from flask import Blueprint, request, jsonify

import traceback
import os
import inspect
import re

from services.embeddings import generate_embeddings
from services.answer_generator import generate_answer
from services.evidence_verifier import verify_evidence

from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import MedCPTReranker

from routes.upload import (
    vector_store,
    bm25_search,
    uploaded_documents
)


# ============================================================
# 1. BLUEPRINT AND INITIALIZATION
# ============================================================

ask_bp = Blueprint("ask", __name__)

hybrid_search = HybridSearch(vector_store, bm25_search)
reranker = MedCPTReranker()

print("\n==============================================", flush=True)
print("ASK FILE LOADED FROM:", os.path.abspath(__file__), flush=True)
print("HYBRID SEARCH METHOD:", inspect.signature(HybridSearch.search), flush=True)
print("ASK API INITIALIZED SUCCESSFULLY", flush=True)
print("==============================================\n", flush=True)


# ============================================================
# 2. CONSTANTS
# ============================================================

QUESTION_WORDS = {
    "what", "when", "where", "why", "who", "whom", "which",
    "how", "does", "do", "did", "is", "are", "was", "were",
    "can", "could", "would", "should", "will", "may", "might",
    "explain", "describe", "define", "compare", "tell", "give",
    "list", "find", "show", "summarize", "discuss", "meaning",
    "name", "paper", "research", "study", "according"
}

VALID_CLAIM_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported"
}

VALID_COMPLETENESS_STATUSES = {
    "complete",
    "partially_complete",
    "insufficient_evidence"
}

RESEARCH_STATUS_MAP = {
    "complete": "sufficient_evidence",
    "partially_complete": "partially_sufficient",
    "insufficient_evidence": "insufficient_evidence",
    "sufficient_evidence": "sufficient_evidence",
    "partially_sufficient": "partially_sufficient",
    "verification_unavailable": "verification_unavailable"
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert scores into JSON-compatible float values."""

    if value is None:
        return None

    try:
        value = float(value)

        if not (float("-inf") < value < float("inf")):
            return None

        return value

    except (TypeError, ValueError):
        return None


def get_named_entities_from_question(question):
    """
    Find likely proper names in a question.

    This is a simple name-presence check, not a general
    semantic relevance detection system.
    """

    words = re.findall(
        r"\b[A-Z][a-zA-Z]{2,}\b",
        question or ""
    )

    entities = [
        word for word in words
        if word.lower() not in QUESTION_WORDS
    ]

    return list(dict.fromkeys(entities))


def default_verification(message):
    """Create a consistent unsupported verification response."""

    return {
        "overall_status": "unsupported",
        "status": "unsupported",
        "claims": [],
        "summary": message,

        "completeness": {
            "status": "insufficient_evidence",
            "missing_findings": [],
            "explanation": message
        },

        "research_completeness": {
            "status": "insufficient_evidence",
            "relevant_sources": [],
            "coverage_gaps": [message],
            "explanation": message
        }
    }


def unavailable_verification(message):
    """Keep API failures separate from unsupported answers."""

    return {
        "overall_status": "verification_unavailable",
        "status": "verification_unavailable",
        "claims": [],
        "summary": (
            "Evidence verification is temporarily unavailable. "
            "Please try again later."
        ),

        "completeness": {
            "status": "insufficient_evidence",
            "missing_findings": [],
            "explanation": message
        },

        "research_completeness": {
            "status": "verification_unavailable",
            "relevant_sources": [],
            "coverage_gaps": [],
            "explanation": message
        },

        "verification_error": message
    }


def clean_unsupported_answer(answer):
    """
    Remove SOURCE labels when the answer has no validated
    supporting evidence.
    """

    if not isinstance(answer, str):
        return str(answer)

    answer = re.sub(
        r"\[\s*SOURCE\b[^\]]*\]",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(r" {2,}", " ", answer)

    return answer.strip()


# ============================================================
# 4. VERIFICATION VALIDATION
# ============================================================

def prepare_verification(
    verification,
    question,
    retrieved_results
):
    """
    Validate and normalize evidence-verifier output.

    Final status considers:

    1. Individual claim support.
    2. Answer completeness.
    3. Research completeness.
    4. Relevant supporting sources.

    This prevents an incomplete answer from being marked
    fully supported simply because all generated claims
    are individually supported.
    """

    if not isinstance(verification, dict):
        verification = {}

    # --------------------------------------------------------
    # STEP 0: Preserve verification-unavailable status
    # --------------------------------------------------------

    incoming_status = str(
        verification.get(
            "overall_status",
            verification.get("status", "")
        )
    ).lower().strip()

    if incoming_status == "verification_unavailable":

        error_message = verification.get(
            "verification_error",
            "The Gemini verification service is temporarily unavailable."
        )

        return unavailable_verification(str(error_message))

    # --------------------------------------------------------
    # STEP 1: Prepare valid source IDs
    # --------------------------------------------------------

    valid_source_ids = {
        f"SOURCE {index}"
        for index in range(1, len(retrieved_results) + 1)
    }

    evidence_text = " ".join(
        str(
            item.get("text")
            or item.get("chunk")
            or ""
        )
        for item in retrieved_results
        if isinstance(item, dict)
    ).lower()

    # --------------------------------------------------------
    # STEP 2: Validate individual claims
    # --------------------------------------------------------

    claims = verification.get("claims", [])

    if not isinstance(claims, list):
        claims = []

    validated_claims = []

    for claim in claims:

        if not isinstance(claim, dict):
            continue

        claim_text = str(
            claim.get("claim")
            or claim.get("text")
            or claim.get("statement")
            or ""
        ).strip()

        if not claim_text:
            continue

        status = str(
            claim.get("status", "unsupported")
        ).lower().strip()

        if status not in VALID_CLAIM_STATUSES:
            status = "unsupported"

        sources = claim.get("supporting_sources", [])

        if not isinstance(sources, list):
            sources = []

        sources = [
            source
            for source in sources
            if isinstance(source, str)
            and source in valid_source_ids
        ]

        explanation = claim.get("explanation", "")

        if not isinstance(explanation, str):
            explanation = ""

        if status in {
            "supported",
            "partially_supported"
        } and not sources:

            status = "unsupported"

            explanation = (
                "No valid retrieved passage was identified "
                "as supporting evidence for this claim."
            )

        if status == "unsupported":
            sources = []

        validated_claims.append({
            **claim,
            "claim": claim_text,
            "status": status,
            "supporting_sources": sources,
            "explanation": explanation
        })

    # --------------------------------------------------------
    # STEP 3: Check named entities
    # --------------------------------------------------------

    named_entities = get_named_entities_from_question(question)

    missing_entities = [
        entity
        for entity in named_entities
        if entity.lower() not in evidence_text
    ]

    entity_mismatch = bool(missing_entities)

    if entity_mismatch:

        entity_message = (
            "The retrieved passages do not contain evidence "
            "about the following named entities: "
            + ", ".join(missing_entities)
            + ". Therefore, the selected papers cannot establish "
              "an answer about these entities."
        )

        for claim in validated_claims:

            claim["status"] = "unsupported"
            claim["supporting_sources"] = []
            claim["explanation"] = entity_message

    # --------------------------------------------------------
    # STEP 4: Validate answer completeness
    # --------------------------------------------------------

    completeness = verification.get("completeness", {})

    if not isinstance(completeness, dict):
        completeness = {}

    completeness_status = str(
        completeness.get(
            "status",
            "insufficient_evidence"
        )
    ).lower().strip()

    if completeness_status not in VALID_COMPLETENESS_STATUSES:
        completeness_status = "insufficient_evidence"

    raw_missing_findings = completeness.get(
        "missing_findings",
        []
    )

    if not isinstance(raw_missing_findings, list):
        raw_missing_findings = []

    valid_missing_findings = []

    for finding in raw_missing_findings:

        if isinstance(finding, str):

            finding_text = finding.strip()

            if finding_text:

                valid_missing_findings.append({
                    "finding": finding_text,
                    "supporting_sources": [],
                    "explanation": ""
                })

            continue

        if not isinstance(finding, dict):
            continue

        finding_text = finding.get("finding", "")

        if (
            not isinstance(finding_text, str)
            or not finding_text.strip()
        ):
            continue

        finding_sources = finding.get(
            "supporting_sources",
            []
        )

        if not isinstance(finding_sources, list):
            finding_sources = []

        finding_sources = [
            source
            for source in finding_sources
            if isinstance(source, str)
            and source in valid_source_ids
        ]

        valid_missing_findings.append({
            **finding,
            "finding": finding_text.strip(),
            "supporting_sources": finding_sources
        })

    # Preserve partially_complete even if Gemini did not
    # provide a structured missing_findings list.

    if entity_mismatch:

        completeness_status = "insufficient_evidence"
        valid_missing_findings = []

        completeness_explanation = (
            "The retrieved passages do not contain the "
            "named entity required to answer the question."
        )

    else:

        completeness_explanation = completeness.get(
            "explanation",
            "Completeness was assessed using the retrieved passages."
        )

        if not isinstance(completeness_explanation, str):
            completeness_explanation = (
                "Completeness was assessed using the retrieved passages."
            )

    completeness = {
        **completeness,
        "status": completeness_status,
        "missing_findings": valid_missing_findings,
        "explanation": completeness_explanation
    }

    # --------------------------------------------------------
    # STEP 5: Validate research completeness
    # --------------------------------------------------------

    research_completeness = verification.get(
        "research_completeness",
        {}
    )

    if not isinstance(research_completeness, dict):
        research_completeness = {}

    relevant_sources = research_completeness.get(
        "relevant_sources",
        []
    )

    if not isinstance(relevant_sources, list):
        relevant_sources = []

    relevant_sources = [
        source
        for source in relevant_sources
        if isinstance(source, str)
        and source in valid_source_ids
    ]

    relevant_sources = list(dict.fromkeys(relevant_sources))

    raw_coverage_gaps = research_completeness.get(
        "coverage_gaps",
        []
    )

    if not isinstance(raw_coverage_gaps, list):
        raw_coverage_gaps = []

    coverage_gaps = [
        gap.strip()
        for gap in raw_coverage_gaps
        if isinstance(gap, str) and gap.strip()
    ]

    raw_research_status = str(
        research_completeness.get(
            "status",
            "insufficient_evidence"
        )
    ).lower().strip()

    research_status = RESEARCH_STATUS_MAP.get(
        raw_research_status,
        "insufficient_evidence"
    )

    # Fallback: use validated claim sources if Gemini
    # did not provide a separate relevant_sources list.

    if not relevant_sources and not entity_mismatch:

        for claim in validated_claims:

            if claim["status"] in {
                "supported",
                "partially_supported"
            }:

                relevant_sources.extend(
                    claim["supporting_sources"]
                )

        relevant_sources = list(
            dict.fromkeys(relevant_sources)
        )

    if entity_mismatch:

        research_status = "insufficient_evidence"
        relevant_sources = []

        coverage_gaps = [
            "The retrieved passages do not contain evidence about: "
            + ", ".join(missing_entities)
        ]

        research_explanation = (
            "The retrieved passages do not contain the named "
            "entity needed to answer this question."
        )

    elif not relevant_sources:

        research_status = "insufficient_evidence"

        coverage_gaps = coverage_gaps or [
            "No relevant supporting passages were identified."
        ]

        research_explanation = (
            "The retrieved passages do not provide sufficient "
            "relevant evidence to answer the question."
        )

    else:

        research_explanation = research_completeness.get(
            "explanation",
            "Relevant passages were identified in the selected papers."
        )

        if not isinstance(research_explanation, str):
            research_explanation = (
                "Relevant passages were identified in the selected papers."
            )

        if (
            research_status == "partially_sufficient"
            and not coverage_gaps
        ):

            coverage_gaps = [
                "Some aspects of the research question are not "
                "covered by the retrieved passages."
            ]

        if (
            research_status == "sufficient_evidence"
            and coverage_gaps
        ):

            research_status = "partially_sufficient"

    research_completeness = {
        **research_completeness,
        "status": research_status,
        "relevant_sources": relevant_sources,
        "coverage_gaps": coverage_gaps,
        "explanation": research_explanation
    }

    # --------------------------------------------------------
    # STEP 6: Calculate final overall status
    # --------------------------------------------------------

    claim_statuses = [
        claim["status"]
        for claim in validated_claims
    ]

    if entity_mismatch:

        overall_status = "unsupported"

    elif (
        not relevant_sources
        or research_status == "insufficient_evidence"
        or not claim_statuses
    ):

        overall_status = "unsupported"

    elif all(
        status == "unsupported"
        for status in claim_statuses
    ):

        overall_status = "unsupported"

    elif any(
        status in {
            "unsupported",
            "partially_supported"
        }
        for status in claim_statuses
    ):

        overall_status = "partially_supported"

    elif (
        completeness_status != "complete"
        or research_status != "sufficient_evidence"
    ):

        overall_status = "partially_supported"

    else:

        overall_status = "supported"

    # --------------------------------------------------------
    # STEP 7: Final consistency checks
    # --------------------------------------------------------

    if overall_status == "unsupported":

        if (
            entity_mismatch
            or not relevant_sources
            or research_status == "insufficient_evidence"
        ):

            for claim in validated_claims:

                claim["status"] = "unsupported"
                claim["supporting_sources"] = []

            completeness["status"] = "insufficient_evidence"
            completeness["missing_findings"] = []

    # --------------------------------------------------------
    # STEP 8: Prepare summary
    # --------------------------------------------------------

    summary = verification.get("summary", "")

    if entity_mismatch:

        summary = (
            "The answer is unsupported by the selected research papers. "
            "The retrieved passages do not contain evidence about "
            + ", ".join(missing_entities)
            + "."
        )

    elif overall_status == "unsupported":

        summary = (
            "The retrieved research passages do not provide "
            "sufficient evidence to support the answer."
        )

    elif overall_status == "partially_supported":

        summary = (
            "Some claims are supported by the retrieved research "
            "passages, but the answer or research coverage is incomplete. "
            "The unsupported or missing aspects should not be treated "
            "as established findings."
        )

    elif not isinstance(summary, str) or not summary.strip():

        summary = (
            "The answer and retrieved research evidence are consistent."
        )

    # --------------------------------------------------------
    # STEP 9: Return normalized verification
    # --------------------------------------------------------

    verification["overall_status"] = overall_status
    verification["status"] = overall_status
    verification["claims"] = validated_claims
    verification["completeness"] = completeness
    verification["research_completeness"] = research_completeness
    verification["summary"] = summary

    # --------------------------------------------------------
    # STEP 10: Debug information
    # --------------------------------------------------------

    print("\n========== FINAL VERIFICATION STATUS ==========", flush=True)

    print("Overall status:", overall_status, flush=True)

    print(
        "Claim statuses:",
        claim_statuses,
        flush=True
    )

    print(
        "Answer completeness:",
        completeness["status"],
        flush=True
    )

    print(
        "Research completeness:",
        research_status,
        flush=True
    )

    print(
        "Relevant sources:",
        relevant_sources,
        flush=True
    )

    print("===============================================\n", flush=True)

    return verification


# ============================================================
# 5. ASK API
# ============================================================

@ask_bp.route("/api/ask", methods=["POST"])
def ask_question():

    try:

        # ----------------------------------------------------
        # STEP 1: Read request
        # ----------------------------------------------------

        data = request.get_json(silent=True) or {}

        question = data.get("question", "")

        if not isinstance(question, str):

            return jsonify({
                "error": "Question must be text."
            }), 400

        question = question.strip()

        selected_document_ids = data.get(
            "selected_document_ids",
            []
        )

        if not question:

            return jsonify({
                "error": "Please enter a research question."
            }), 400

        if not isinstance(selected_document_ids, list):

            return jsonify({
                "error": "Selected document IDs must be a list."
            }), 400

        if not selected_document_ids:

            return jsonify({
                "error": "Please select at least one research paper."
            }), 400

        print("\n========== NEW RESEARCH QUESTION ==========", flush=True)
        print("Question:", question, flush=True)
        print("Selected documents:", selected_document_ids, flush=True)

        # ----------------------------------------------------
        # STEP 2: Find selected documents
        # ----------------------------------------------------

        selected_id_strings = {
            str(document_id)
            for document_id in selected_document_ids
        }

        selected_documents = [
            document
            for document in uploaded_documents
            if str(document.get("document_id"))
            in selected_id_strings
        ]

        if not selected_documents:

            return jsonify({
                "error": "Selected research papers were not found."
            }), 400

        valid_document_ids = {
            str(document.get("document_id"))
            for document in selected_documents
        }

        # ----------------------------------------------------
        # STEP 3: Generate query embedding
        # ----------------------------------------------------

        print("[ASK] Generating query embedding...", flush=True)

        query_embedding = generate_embeddings([question])[0]

        # ----------------------------------------------------
        # STEP 4: Hybrid search
        # ----------------------------------------------------

        print("[ASK] Starting hybrid search...", flush=True)

        search_results = hybrid_search.search(
            query=question,
            query_embedding=query_embedding,
            top_k=30,
            selected_document_ids=selected_document_ids
        )

        if not search_results:
            search_results = []

        print(
            "[ASK] Hybrid search results:",
            len(search_results),
            flush=True
        )

        # ----------------------------------------------------
        # STEP 5: Filter selected documents
        # ----------------------------------------------------

        filtered_results = []

        for result in search_results:

            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            document_id = metadata.get("document_id")

            if str(document_id) in valid_document_ids:
                filtered_results.append(result)

        print(
            "[ASK] Filtered results:",
            len(filtered_results),
            flush=True
        )

        if not filtered_results:

            message = (
                "No passages were retrieved from the selected papers."
            )

            verification = default_verification(message)

            return jsonify({

                "message": "No relevant passages were found.",

                "question": question,

                "answer": (
                    "The selected research papers do not contain "
                    "sufficient retrieved information to answer "
                    "this question."
                ),

                "verification": verification,

                "selected_document_ids": selected_document_ids,

                "retrieved_count": 0,

                "retrieved_results": []

            }), 200

        # ----------------------------------------------------
        # STEP 6: Rerank retrieved passages
        # ----------------------------------------------------

        print("[ASK] Starting MedCPT reranking...", flush=True)

        reranked_results = reranker.rerank(
            query=question,
            documents=filtered_results,
            top_k=5
        )

        if not reranked_results:
            reranked_results = []

        print(
            "[ASK] Reranked results:",
            len(reranked_results),
            flush=True
        )

        if not reranked_results:

            message = (
                "No passages were available after reranking."
            )

            verification = default_verification(message)

            return jsonify({

                "message": "No relevant passages were found.",

                "question": question,

                "answer": (
                    "No sufficiently relevant passages were found "
                    "in the selected research papers."
                ),

                "verification": verification,

                "selected_document_ids": selected_document_ids,

                "retrieved_count": 0,

                "retrieved_results": []

            }), 200

        # ----------------------------------------------------
        # STEP 7: Format retrieved passages
        # ----------------------------------------------------

        retrieved_results = []

        for index, result in enumerate(
            reranked_results,
            start=1
        ):

            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            chunk_text_value = (
                result.get("chunk")
                or result.get("text")
                or result.get("chunk_text")
                or ""
            )

            if chunk_text_value is None:
                chunk_text_value = ""

            rerank_score = safe_float(
                result.get("rerank_score")
            )

            raw_rerank_score = safe_float(
                result.get(
                    "rerank_logit",
                    result.get("medcpt_raw_score")
                )
            )

            retrieved_results.append({

                "rank": index,

                "chunk": str(chunk_text_value),

                "text": str(chunk_text_value),

                "filename": metadata.get(
                    "filename",
                    "Unknown paper"
                ),

                "document_id": metadata.get("document_id"),

                "chunk_index": metadata.get("chunk_index"),

                "rrf_score": safe_float(
                    result.get("rrf_score")
                ),

                "rerank_score": rerank_score,

                "reranker_score": rerank_score,

                "rerank_logit": raw_rerank_score,

                "medcpt_raw_score": raw_rerank_score

            })

        if not retrieved_results:

            verification = default_verification(
                "No valid passages were available after reranking."
            )

            return jsonify({

                "message": "No valid passages were found.",

                "question": question,

                "answer": (
                    "The selected research papers do not contain "
                    "sufficient retrieved information to answer "
                    "this question."
                ),

                "verification": verification,

                "selected_document_ids": selected_document_ids,

                "retrieved_count": 0,

                "retrieved_results": []

            }), 200

        # ----------------------------------------------------
        # STEP 8: Generate answer using Gemini
        # ----------------------------------------------------

        print("[ASK] Generating answer using Gemini...", flush=True)

        answer = generate_answer(
            question,
            retrieved_results
        )

        if not isinstance(answer, str):
            answer = str(answer)

        print("[ASK] Answer generation completed.", flush=True)

        # ----------------------------------------------------
        # STEP 9: Verify answer using evidence verifier
        # ----------------------------------------------------

        print("[ASK] Starting evidence verification...", flush=True)

        try:

            raw_verification = verify_evidence(
                question,
                answer,
                retrieved_results
            )

            print(
                "[DEBUG] RAW VERIFICATION:",
                raw_verification,
                flush=True
            )

            verification = prepare_verification(
                raw_verification,
                question,
                retrieved_results
            )

            print(
                "[ASK] Evidence verification completed.",
                flush=True
            )

        except Exception as verification_error:

            print(
                "[ASK] EVIDENCE VERIFICATION ERROR:",
                repr(verification_error),
                flush=True
            )

            verification = unavailable_verification(
                str(verification_error)
            )

        # ----------------------------------------------------
        # STEP 10: Remove unsupported source citations
        # ----------------------------------------------------

        research_status = verification.get(
            "research_completeness",
            {}
        ).get("status", "insufficient_evidence")

        if research_status == "insufficient_evidence":

            answer = clean_unsupported_answer(answer)

        # ----------------------------------------------------
        # STEP 11: Prepare API response
        # ----------------------------------------------------

        response_data = {

            "message": (
                "Answer generated, but evidence verification "
                "is temporarily unavailable."
                if verification.get("overall_status")
                == "verification_unavailable"
                else "Answer generated successfully!"
            ),

            "question": question,

            "answer": answer,

            "verification": verification,

            "selected_document_ids": selected_document_ids,

            "retrieved_count": len(retrieved_results),

            "retrieved_results": retrieved_results

        }

        # ----------------------------------------------------
        # STEP 12: Print debugging information
        # ----------------------------------------------------

        print("\n========== API RESPONSE SUMMARY ==========", flush=True)

        print(
            "ASK FILE:",
            os.path.abspath(__file__),
            flush=True
        )

        print("Question:", question, flush=True)

        print(
            "Overall status:",
            verification["overall_status"],
            flush=True
        )

        print(
            "Answer completeness:",
            verification["completeness"]["status"],
            flush=True
        )

        print(
            "Research completeness:",
            verification["research_completeness"]["status"],
            flush=True
        )

        print(
            "Relevant sources:",
            verification["research_completeness"]["relevant_sources"],
            flush=True
        )

        print(
            "Claims:",
            [
                claim["status"]
                for claim in verification["claims"]
            ],
            flush=True
        )

        print(
            "Retrieved passages:",
            len(retrieved_results),
            flush=True
        )

        print("==========================================\n", flush=True)

        return jsonify(response_data), 200

    except Exception as error:

        print(
            "\n========== FULL ERROR TRACEBACK ==========",
            flush=True
        )

        traceback.print_exc()

        print(
            "===========================================\n",
            flush=True
        )

        return jsonify({

            "error": "Question processing failed: " + str(error)

        }), 500
