
import os

from dotenv import load_dotenv
from google import genai

from services.gemini_retry import call_gemini_with_retry


# ==============================================
# GEMINI CONFIGURATION
# ==============================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# ==============================================
# ANSWER GENERATION
# ==============================================

def generate_answer(question, retrieved_results):

    """
    Generate research answers using only
    the supplied retrieved research passages.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_results:
        return (
            "The available retrieved research passages "
            "do not provide sufficient evidence to answer "
            "this question."
        )

    # ------------------------------------------
    # PREPARE RESEARCH EVIDENCE
    # ------------------------------------------

    evidence_parts = []

    for index, result in enumerate(
        retrieved_results,
        start=1
    ):

        filename = result.get(
            "filename",
            "Unknown paper"
        )

        chunk_index = result.get(
            "chunk_index",
            "Unknown"
        )

        passage = (
            result.get("chunk")
            or result.get("text")
            or ""
        )

        if not isinstance(passage, str) or not passage.strip():
            continue

        evidence_parts.append(
            f"""
[SOURCE {index}]
Paper: {filename}
Chunk: {chunk_index}

Passage:
{passage}
"""
        )

    if not evidence_parts:
        return (
            "The retrieved results do not contain "
            "usable research-paper passages."
        )

    evidence_text = "\n".join(evidence_parts)

    # ------------------------------------------
    # CONSTRUCT GEMINI PROMPT
    # ------------------------------------------

    prompt = f"""
You are an expert scientific research assistant.

Your responsibility is to answer research questions
using ONLY the supplied research-paper passages.

USER QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{evidence_text}

IMPORTANT INSTRUCTIONS:

1. Read all retrieved passages carefully.

2. Identify distinct findings that directly answer
   the user's question.

3. If the question asks for limitations, identify
   limitations explicitly discussed or directly
   supported by the passages.

4. If the question asks for advantages, identify
   advantages supported by the passages.

5. Do not repeat the same finding using different words.

6. Do not invent scientific facts, limitations,
   advantages, or explanations.

7. Do not use outside knowledge.

8. Every factual finding must have at least one
   correct source reference.

9. Use the exact source IDs provided above.

10. If multiple sources support the same finding,
    include all relevant source references.

11. If only one finding is supported, report only one.

12. Never generate additional findings just to
    increase answer length.

13. Clearly distinguish supported information from
    information that is not established by the passages.

14. If evidence is insufficient, explain this clearly.

15. Write in simple, understandable English.

16. Do not create source references that are not
    present in the retrieved evidence.

17. Return clean Markdown formatting.
    Do not escape Markdown characters with backslashes.

REQUIRED ANSWER FORMAT:

## Research answer

Give a short introduction to the answer.

## Key findings

Present each distinct finding as a numbered point.

For each finding:

- Give a meaningful heading.
- Explain the finding in simple English.
- Include the supporting source references.

## Conclusion

Summarize what the retrieved research evidence establishes.

IMPORTANT:
Accuracy and research evidence are more important
than the number of findings.
"""

    # ------------------------------------------
    # GENERATE ANSWER USING GEMINI
    # ------------------------------------------

    response = call_gemini_with_retry(
        lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    )

    if not response or not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text.strip()
