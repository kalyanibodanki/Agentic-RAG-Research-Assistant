# Agentic RAG – Scientific Research Paper Assistant

An intelligent research paper assistant that uses Hybrid Search, MedCPT Reranking, Gemini-based Answer Generation, and Evidence Verification to provide evidence-grounded answers from scientific research papers.

## Project Overview

The Agentic RAG Scientific Research Paper Assistant is designed to help users explore and understand scientific research papers efficiently.

Traditional Retrieval-Augmented Generation (RAG) systems may retrieve relevant documents but can struggle with complex questions, passage relevance, and unsupported generated answers.

To address these limitations, this project implements an enhanced Agentic RAG pipeline combining keyword-based retrieval, semantic vector search, neural reranking, LLM-based answer generation, and evidence verification.

Users can upload research papers in PDF format and ask questions about their contents. The system retrieves relevant passages, generates answers using Google Gemini, and checks whether the generated claims are supported by the retrieved evidence.

## Project Objectives

- Simplify scientific research paper exploration.
- Retrieve relevant information from uploaded PDF documents.
- Improve retrieval performance using Hybrid Search.
- Improve passage relevance using MedCPT Reranking.
- Generate evidence-grounded answers using Gemini.
- Verify generated claims against retrieved research evidence.
- Identify unsupported claims and incomplete answers.

## Key Features

### 1. PDF Upload and Processing
- Upload scientific research papers in PDF format.
- Extract text from PDF documents.
- Divide extracted text into smaller chunks.
- Generate embeddings for document chunks.

### 2. Hybrid Search
- Combines semantic vector search and BM25 keyword search.
- Retrieves relevant passages using both retrieval techniques.

### 3. MedCPT Reranking
- Uses a biomedical cross-encoder model.
- Reorders retrieved passages based on query-passage relevance.

### 4. AI-Based Answer Generation
- Integrates the Google Gemini API.
- Generates answers based on retrieved research-paper passages.

### 5. Evidence Verification
- Checks whether generated claims are supported by retrieved evidence.
- Identifies supported, partially supported, and unsupported claims.
- Evaluates answer completeness.
- Provides source references for supporting passages.

### 6. Interactive Web Interface
- React-based frontend.
- Allows users to upload PDFs and ask research questions.
- Displays generated answers and supporting sources.

## System Architecture

The application follows a React frontend and Flask backend architecture.

### Workflow

1. User uploads a research paper in PDF format.
2. Flask extracts text from the uploaded document.
3. Extracted text is divided into chunks.
4. Sentence Transformers generates embeddings.
5. Document chunks are indexed using FAISS and BM25.
6. User submits a research question.
7. Hybrid Search retrieves relevant passages.
8. MedCPT Reranker reorders the retrieved passages.
9. Gemini generates an answer using retrieved evidence.
10. Evidence Verification checks the generated answer.
11. The frontend displays the answer, evidence, and source information.

### Architecture Diagram

```text
                 USER
                   |
                   v
             REACT FRONTEND
                   |
                   v
              FLASK BACKEND
                   |
                   v
          PDF UPLOAD AND PROCESSING
                   |
                   v
             TEXT EXTRACTION
                   |
                   v
              TEXT CHUNKING
                   |
                   v
           EMBEDDING GENERATION
                   |
             +-----+-----+
             |           |
             v           v
            FAISS       BM25
        Vector Search  Keyword Search
             |           |
             +-----+-----+
                   |
                   v
              HYBRID SEARCH
                   |
                   v
             MEDCPT RERANKER
                   |
                   v
          GEMINI ANSWER GENERATION
                   |
                   v
           EVIDENCE VERIFICATION
                   |
                   v
             ANSWER AND SOURCES
                   |
                   v
              USER INTERFACE
```

## Technologies Used

| Category | Technologies |
|---|---|
| Frontend | React, JavaScript/TypeScript, HTML, CSS |
| Backend | Python, Flask, Flask-CORS |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Embeddings | Sentence Transformers |
| Reranking | MedCPT Cross-Encoder |
| LLM | Google Gemini API |
| PDF Processing | PyPDF |
| Numerical Processing | NumPy |
| Development | Visual Studio Code, Google Colab |
| Version Control | Git, GitHub |
| Frontend Hosting | Vercel |

## Project Structure

```text
Agentic_RAG_Research_Assistant/
|
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|
|-- backend/
|   |-- app.py
|   |-- routes/
|   |-- services/
|   |-- retrieval/
|   |-- requirements.txt
|   |-- .env
|
|-- README.md
```

Note: This is a simplified representation of the project structure. Refer to the actual repository for all files and folders.

## Installation and Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Git
- Visual Studio Code
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Agentic_RAG_Research_Assistant
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv311
```

Activate the environment on Windows PowerShell:

```powershell
.\venv311\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API Key

Create a `.env` file inside the backend directory.

Add:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your actual Gemini API key.

**Important:** Never upload your `.env` file or API key to GitHub.

### 4. Start the Backend

Make sure the virtual environment is activated.

Run:

```bash
python app.py
```

The backend runs locally at:

```text
http://127.0.0.1:5000
```

Health endpoint:

```text
http://127.0.0.1:5000/api/health
```

### 5. Frontend Setup

Open a separate terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm run dev
```

Open the local frontend URL displayed in the terminal.

Note: The frontend must be configured to communicate with the running Flask backend. The Vercel frontend alone does not run the Flask API.

## Retrieval Evaluation

The retrieval pipeline was evaluated using 100 queries.

### Evaluation Results

| Retrieval Method | Recall@10 | MRR@10 |
|---|---:|---:|
| Vector Search | 0.8160 | 0.6682 |
| BM25 Search | 0.8603 | 0.6979 |
| Hybrid Search | 0.8922 | 0.7464 |
| Hybrid + Initial Reranker | 0.8510 | 0.6926 |
| Hybrid + Alternative Reranker | 0.8930 | 0.7698 |

### Evaluation Metrics

**Recall@10:** Measures the proportion of relevant documents retrieved within the top 10 results.

**MRR@10:** Measures the ranking position of the first relevant result across queries.

The reported results show that Hybrid Search improved retrieval performance compared with individual vector and BM25 search. The alternative reranker improved the reported MRR@10.

Note: These metrics represent retrieval performance on the tested dataset and queries. They do not directly measure the factual accuracy of generated answers.

## Deployment Status

### Frontend
The React frontend is deployed using Vercel.

**Frontend URL:** ADD_YOUR_VERCEL_URL_HERE

### Backend
The Flask backend currently runs locally on the developer's machine.

For complete application functionality, the backend must be running and accessible to the frontend.

Permanent cloud deployment of the backend is planned as a future enhancement.

## Current Limitations

- The backend is not permanently deployed to a cloud environment.
- The application depends on Gemini API availability and usage quotas.
- Retrieval quality depends on the uploaded documents and their content.
- Hybrid Search may not retrieve every relevant passage for complex questions.
- Reranking cannot recover passages that were not retrieved initially.
- Evidence verification reduces unsupported claims but cannot guarantee complete factual correctness.
- Evaluation is currently limited to the tested dataset and queries.
- Uploaded documents and in-memory indexes are not guaranteed to persist after application restarts.
- The current implementation is not designed for large-scale concurrent usage.

## Future Enhancements

- Deploy the Flask backend to a scalable cloud environment.
- Implement persistent storage for uploaded documents and indexes.
- Introduce user authentication and authorization.
- Improve support for concurrent users.
- Evaluate the system on larger and more diverse scientific datasets.
- Improve source traceability and evidence verification.
- Explore advanced retrieval and reranking techniques.

## Author

**BODANKI KALYANI**

Master of Computer Applications (MCA)

Acharya Nagarjuna University

## License

This project is developed for academic and educational purposes.