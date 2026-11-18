
import os
import json
import re

from dotenv import load_dotenv
from google import genai

from services.gemini_retry import call_gemini_with_retry


# ==================================================
# 1. GEMINI CONFIGURATION
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"

ALLOWED_CLAIM_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported"
}

ALLOWED_COMPLETENESS = {
    "complete",
    "partially_complete",
    "insufficient_evidence"
}


# ==================================================
# 2. JSON EXTRACTION
# ==================================================

def extract_json(response_text):

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    response_text = response_text.strip()

    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(r"\s*```$", "", response_text)

    try:
        return json.loads(response_text)

    except json.JSONDecodeError:

        start = response_text.find("{")
        end = response_text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("Gemini did not return valid JSON.")

        return json.loads(response_text[start:end + 1])


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def safe_string(value, default=""):

    if isinstance(value, str):
        return value.strip()

    return default


def normalize_source_ids(sources, valid_source_ids):
    """
    Normalize source references such as:
    SOURCE 1
    [SOURCE 1]
    source_1

    Only existing source IDs are accepted.
    """

    if not isinstance(sources, list):
        return []

    normalized = []

    for source in sources:

        if not isinstance(source, str):
            continue

        match = re.search(
            r"SOURCE[\s_\-]*(\d+)",
            source,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        source_id = f"SOURCE {int(match.group(1))}"

        if source_id in valid_source_ids:
            normalized.append(source_id)

    return list(dict.fromkeys(normalized))


def get_result_metadata(result):

    metadata = result.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    metadata = dict(metadata)

    for key in (
        "filename",
        "document_id",
        "chunk_index",
        "page_number"
    ):

        if key not in metadata and key in result:
            metadata[key] = result[key]

    return metadata


def create_default_verification(message):

    return {
        "overall_status": "unsupported",
        "status": "unsupported",
        "claims": [],

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
        },

        "summary": message
    }


def create_unavailable_verification(error):

    error_message = str(error)

    print(
        "\n[VERIFICATION ERROR]",
        error_message,
        flush=True
    )

    return {
        "overall_status": "verification_unavailable",
        "status": "verification_unavailable",
        "claims": [],

        "completeness": {
            "status": "insufficient_evidence",
            "missing_findings": [],
            "explanation": (
                "Evidence verification could not be completed "
                "because the verification service encountered an error."
            )
        },

        "research_completeness": {
            "status": "verification_unavailable",
            "relevant_sources": [],
            "coverage_gaps": [],
            "explanation": (
                "The verification service is temporarily unavailable."
            )
        },

        "verification_error": error_message,

        "summary": (
            "Verification is temporarily unavailable. "
            "Please try again later."
        )
    }


# ==================================================
# 4. MAIN EVIDENCE VERIFICATION
# ==================================================

def verify_evidence(question, answer, retrieved_results):

    print(
        "\n========== EVIDENCE VERIFICATION STARTED ==========",
        flush=True
    )

    print("Question:", question, flush=True)

    print(
        "Retrieved passages:",
        len(retrieved_results or []),
        flush=True
    )

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Research question is empty.")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generated answer is empty.")

    if not retrieved_results:

        return create_default_verification(
            "No retrieved research evidence was available."
        )

    # --------------------------------------------------
    # STEP 1: PREPARE EVIDENCE
    # --------------------------------------------------

    evidence_parts = []
    valid_source_ids = set()

    for result in retrieved_results:

        if not isinstance(result, dict):
            continue

        passage = (
            result.get("chunk")
            or result.get("text")
            or ""
        )

        if not isinstance(passage, str) or not passage.strip():
            continue

        source_id = f"SOURCE {len(evidence_parts) + 1}"

        metadata = get_result_metadata(result)

        filename = metadata.get(
            "filename",
            "Unknown paper"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        document_id = metadata.get(
            "document_id",
            "Unknown"
        )

        valid_source_ids.add(source_id)

        evidence_parts.append(
            f"""
[{source_id}]
Paper: {filename}
Document ID: {document_id}
Chunk: {chunk_index}

Passage:
{passage.strip()}
"""
        )

    if not evidence_parts:

        return create_default_verification(
            "Retrieved results did not contain readable passages."
        )

    evidence_text = "\n".join(evidence_parts)

    # --------------------------------------------------
    # STEP 2: VERIFICATION PROMPT
    # --------------------------------------------------

    prompt = f"""
You are an expert scientific research evidence evaluator.

Your job is to evaluate a generated answer using ONLY
the retrieved research passages.

Do not use outside knowledge.
Do not invent findings.
Do not reject relevant evidence simply because
the wording differs from the generated answer.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED RESEARCH PASSAGES:
{evidence_text}


==================================================
PART 1: IDENTIFY RELEVANT SOURCES
==================================================

Read the question carefully.

For every retrieved passage, decide whether it provides
information that helps answer the question.

IMPORTANT:

1. Judge semantic meaning, not exact word matching.
2. A passage may be relevant even if it uses different
   terminology from the question.
3. A passage may support only one aspect of a broad question.
4. Do not reject a passage just because it does not answer
   every part of the question.
5. Do not mark unrelated background information as relevant.

Return the IDs of all passages that provide direct or
partial evidence for the question.

For example, if SOURCE 2, SOURCE 3 and SOURCE 4 are relevant:

"relevant_sources": [
    "SOURCE 2",
    "SOURCE 3",
    "SOURCE 4"
]

Use the exact source labels shown in the retrieved passages.

==================================================
PART 2: VERIFY ANSWER CLAIMS
==================================================

Break the generated answer into individual factual claims.

For every claim:

supported:
The retrieved passages directly establish the claim.

partially_supported:
The passages establish some, but not all, of the claim.

unsupported:
The retrieved passages do not establish the claim.

Only cite sources that actually support that claim.

A source may be relevant to the question without
supporting every claim in the answer.

Do not use unrelated sources to support claims.

==================================================
PART 3: ANSWER COMPLETENESS
==================================================

Check whether the answer covers the important findings
that are present in the relevant retrieved passages.

complete:
The answer covers the important relevant findings.

partially_complete:
Some relevant findings are missing.

insufficient_evidence:
The retrieved passages do not provide enough information
to assess the answer.

For partially_complete, list the missing findings and
the sources that establish those findings.

==================================================
PART 4: RESEARCH COMPLETENESS
==================================================

Evaluate how well the retrieved passages cover the
question's scope.

complete:
The retrieved passages provide sufficiently broad
evidence for the question.

partially_complete:
Relevant evidence exists, but some aspects are missing.

insufficient_evidence:
The passages are unrelated or provide too little evidence.

Do not claim that the entire scientific literature
has been reviewed.

Evaluate only the passages provided.

==================================================
IMPORTANT EXAMPLE
==================================================

Question:
What are the limitations of traditional RAG systems?

Passage:
"Standard RAG pipelines struggle with long-tail knowledge
and can suffer from hallucinations when retrieval fails."

This passage IS RELEVANT.

Passage:
"Static retrieve-then-generate paradigms lack flexibility
for complex multi-hop queries."

This passage IS RELEVANT.

Passage:
"Flattening documents into disjointed chunks discards
headings and document boundaries."

This passage IS RELEVANT.

Do not mark these passages unsupported merely because
the generated answer uses different wording.

==================================================
RETURN ONLY VALID JSON
==================================================

Use exactly this structure:

{{
    "overall_status": "partially_supported",

    "claims": [
        {{
            "claim": "A factual claim from the generated answer.",
            "status": "supported",
            "supporting_sources": ["SOURCE 1"],
            "explanation": "The passage directly establishes this claim."
        }}
    ],

    "completeness": {{
        "status": "complete",
        "missing_findings": [],
        "explanation": "The answer covers the important relevant findings."
    }},

    "research_completeness": {{
        "status": "partially_complete",
        "relevant_sources": ["SOURCE 1", "SOURCE 2"],
        "coverage_gaps": [],
        "explanation": "The retrieved passages provide relevant evidence."
    }},

    "summary": "The answer was evaluated against the retrieved research passages."
}}

Allowed claim statuses:
supported, partially_supported, unsupported.

Allowed completeness statuses:
complete, partially_complete, insufficient_evidence.

Remember:
- Use exact source labels.
- Do not invent source IDs.
- Return relevant sources even when only partially relevant.
- Return JSON only.
"""

    # --------------------------------------------------
    # STEP 3: CALL GEMINI
    # --------------------------------------------------

    try:

        response = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
        )

        if not response or not response.text:

            raise ValueError(
                "Gemini returned an empty verification response."
            )

        print(
            "\n[DEBUG] GEMINI RAW VERIFICATION RESPONSE:",
            response.text,
            flush=True
        )

        verification = extract_json(response.text)

        if not isinstance(verification, dict):

            raise ValueError(
                "Gemini returned an invalid JSON object."
            )

    except Exception as error:

        return create_unavailable_verification(error)

    # --------------------------------------------------
    # STEP 4: VALIDATE RELEVANT SOURCES
    # --------------------------------------------------

    raw_research = verification.get(
        "research_completeness",
        {}
    )

    if not isinstance(raw_research, dict):
        raw_research = {}

    relevant_sources = normalize_source_ids(
        raw_research.get("relevant_sources", []),
        valid_source_ids
    )

    research_status = raw_research.get("status")

    if research_status not in ALLOWED_COMPLETENESS:
        research_status = "insufficient_evidence"

    raw_gaps = raw_research.get("coverage_gaps", [])

    if not isinstance(raw_gaps, list):
        raw_gaps = []

    coverage_gaps = [
        gap.strip()
        for gap in raw_gaps
        if isinstance(gap, str) and gap.strip()
    ]

    research_explanation = safe_string(
        raw_research.get("explanation")
    )

    if not relevant_sources:

        research_status = "insufficient_evidence"

        coverage_gaps = [
            "No retrieved passages were identified as relevant to the question."
        ]

        research_explanation = (
            "The retrieved passages were not identified as "
            "direct or partial evidence for the question."
        )

    elif research_status == "complete" and coverage_gaps:

        research_status = "partially_complete"

    elif research_status == "partially_complete" and not coverage_gaps:

        coverage_gaps = [
            "Some aspects of the research question may not be covered."
        ]

    if not research_explanation:

        research_explanation = (
            "Research completeness was assessed using retrieved passages."
        )

    # --------------------------------------------------
    # STEP 5: VALIDATE CLAIMS
    # --------------------------------------------------

    raw_claims = verification.get("claims", [])

    if not isinstance(raw_claims, list):
        raw_claims = []

    validated_claims = []

    for claim in raw_claims:

        if not isinstance(claim, dict):
            continue

        claim_text = safe_string(claim.get("claim"))

        if not claim_text:
            continue

        status = claim.get("status")

        if status not in ALLOWED_CLAIM_STATUSES:
            status = "unsupported"

        sources = normalize_source_ids(
            claim.get("supporting_sources", []),
            valid_source_ids
        )

        explanation = safe_string(
            claim.get("explanation")
        )

        # Supporting sources must also be relevant
        # to the question.

        sources = [
            source
            for source in sources
            if source in relevant_sources
        ]

        if status in {"supported", "partially_supported"} and not sources:

            status = "unsupported"

            explanation = (
                "No valid relevant source was identified "
                "for this claim."
            )

        if status == "unsupported":
            sources = []

        validated_claims.append({
            "claim": claim_text,
            "status": status,
            "supporting_sources": sources,
            "explanation": explanation
        })

    # --------------------------------------------------
    # STEP 6: CALCULATE OVERALL CLAIM STATUS
    # --------------------------------------------------

    if not relevant_sources or not validated_claims:

        overall_status = "unsupported"

    else:

        statuses = [
            claim["status"]
            for claim in validated_claims
        ]

        if all(status == "supported" for status in statuses):

            overall_status = "supported"

        elif all(status == "unsupported" for status in statuses):

            overall_status = "unsupported"

        else:

            overall_status = "partially_supported"

    # --------------------------------------------------
    # STEP 7: VALIDATE ANSWER COMPLETENESS
    # --------------------------------------------------

    raw_completeness = verification.get(
        "completeness",
        {}
    )

    if not isinstance(raw_completeness, dict):
        raw_completeness = {}

    completeness_status = raw_completeness.get("status")

    if completeness_status not in ALLOWED_COMPLETENESS:
        completeness_status = "insufficient_evidence"

    raw_missing = raw_completeness.get(
        "missing_findings",
        []
    )

    if not isinstance(raw_missing, list):
        raw_missing = []

    missing_findings = []

    for finding in raw_missing:

        if not isinstance(finding, dict):
            continue

        finding_text = safe_string(
            finding.get("finding")
        )

        if not finding_text:
            continue

        finding_sources = normalize_source_ids(
            finding.get("supporting_sources", []),
            valid_source_ids
        )

        finding_sources = [
            source
            for source in finding_sources
            if source in relevant_sources
        ]

        if not finding_sources:
            continue

        missing_findings.append({
            "finding": finding_text,
            "supporting_sources": finding_sources,
            "explanation": safe_string(
                finding.get("explanation")
            )
        })

    if not relevant_sources:

        completeness_status = "insufficient_evidence"
        missing_findings = []

        completeness_explanation = (
            "The retrieved passages do not contain enough "
            "relevant information to assess answer completeness."
        )

    else:

        if completeness_status == "complete" and missing_findings:

            completeness_status = "partially_complete"

        completeness_explanation = safe_string(
            raw_completeness.get("explanation"),
            "Answer completeness was evaluated using relevant passages."
        )

    completeness = {
        "status": completeness_status,
        "missing_findings": missing_findings,
        "explanation": completeness_explanation
    }

    # --------------------------------------------------
    # STEP 8: FINAL RESULT
    # --------------------------------------------------

    if not relevant_sources:

        summary = (
            "The retrieved passages do not provide sufficient "
            "relevant evidence to verify the answer."
        )

    else:

        summary = safe_string(
            verification.get("summary"),
            "Verification completed using retrieved research passages."
        )

    result = {
        "overall_status": overall_status,
        "status": overall_status,
        "claims": validated_claims,
        "completeness": completeness,

        "research_completeness": {
            "status": research_status,
            "relevant_sources": relevant_sources,
            "coverage_gaps": coverage_gaps,
            "explanation": research_explanation
        },

        "summary": summary
    }

    # --------------------------------------------------
    # STEP 9: DEBUG INFORMATION
    # --------------------------------------------------

    print(
        "\n========== VERIFICATION COMPLETED ==========",
        flush=True
    )

    print("Overall status:", overall_status, flush=True)

    print(
        "Answer completeness:",
        completeness_status,
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

    print(
        "Claims checked:",
        len(validated_claims),
        flush=True
    )

    print("============================================\n", flush=True)

    return result
