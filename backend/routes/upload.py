
import os
import uuid

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from pypdf import PdfReader

from services.chunking import chunk_text
from services.embeddings import generate_embeddings

from retrieval.vector_store import VectorStore
from retrieval.bm25_search import BM25Search


upload_bp = Blueprint("upload", __name__)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Shared FAISS and BM25 instances
vector_store = VectorStore(dimension=384)
bm25_search = BM25Search()

# Stores uploaded document information
uploaded_documents = []


@upload_bp.route("/api/upload", methods=["POST"])
def upload_pdf():

    file_path = None

    if "file" not in request.files:
        return jsonify({
            "error": "No PDF file was provided."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Please select a PDF file."
        }), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "error": "Only PDF files are allowed."
        }), 400

    original_filename = secure_filename(file.filename)

    if not original_filename:
        return jsonify({
            "error": "Invalid filename."
        }), 400

    document_id = str(uuid.uuid4())

    unique_filename = f"{document_id}_{original_filename}"

    file_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    try:
        # STEP 1: Save PDF
        file.save(file_path)

        # STEP 2: Read PDF
        reader = PdfReader(file_path)

        if reader.is_encrypted:
            raise ValueError(
                "Encrypted PDFs are not supported."
            )

        # STEP 3: Extract text
        extracted_text = ""

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):
            page_text = page.extract_text() or ""

            extracted_text += (
                f"\n--- Page {page_number} ---\n"
                f"{page_text}\n"
            )

        if not extracted_text.strip():
            raise ValueError(
                "No readable text was found in this PDF."
            )

        # STEP 4: Generate chunks
        chunks = chunk_text(
            extracted_text,
            chunk_size=1000,
            overlap=200
        )

        if not chunks:
            raise ValueError(
                "No text chunks were generated."
            )

        # STEP 5: Generate embeddings
        embeddings = generate_embeddings(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        if embeddings.shape[1] != 384:
            raise ValueError(
                "Unexpected embedding dimension."
            )

        # STEP 6: Create metadata
        metadatas = []

        for chunk_index in range(len(chunks)):
            metadatas.append({
                "document_id": document_id,
                "filename": original_filename,
                "chunk_index": chunk_index
            })

        # STEP 7: Add documents to BM25
        bm25_search.add_documents(
            chunks,
            metadatas
        )

        # STEP 8: Add documents to FAISS
        vector_store.add_documents(
            chunks,
            embeddings,
            metadatas
        )

        # STEP 9: Store document information
        page_count = len(reader.pages)
        total_chunks = len(chunks)

        document_info = {
            "document_id": document_id,
            "filename": original_filename,

            # Names used by the frontend
            "page_count": page_count,
            "pages": page_count,

            "total_chunks": total_chunks,
            "chunks_count": total_chunks
        }

        uploaded_documents.append(document_info)

        # STEP 10: Return upload response
        return jsonify({
            "message": (
                "PDF uploaded, processed, and indexed successfully!"
            ),

            "document": document_info,

            # Frontend-compatible fields
            "filename": original_filename,
            "pages": page_count,
            "chunks_created": total_chunks,
            "extracted_preview": extracted_text[:1000],
            "chunks": chunks[:3],

            # Original response fields
            "text_length": len(extracted_text),
            "embedding_dimension": int(embeddings.shape[1]),
            "indexed_chunks": int(vector_store.index.ntotal),
            "bm25_indexed_chunks": len(bm25_search.chunks),
            "text_preview": extracted_text[:1000],
            "chunks_preview": chunks[:3]

        }), 200

    except Exception as error:

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        print("PDF Upload Error:", repr(error))

        return jsonify({
            "error": f"Could not process PDF: {str(error)}"
        }), 400


@upload_bp.route("/api/documents", methods=["GET"])
def get_uploaded_documents():

    return jsonify({
        "documents": uploaded_documents
    }), 200
