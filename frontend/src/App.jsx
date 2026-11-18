
import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import DeveloperDashboard from "./DeveloperDashboard";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

// --------------------------------------------------
// STATUS HELPERS
// --------------------------------------------------

function normalizeStatus(value) {
  return String(value || "not_performed")
    .toLowerCase()
    .replaceAll("-", "_")
    .replaceAll(" ", "_");
}

function isInsufficientEvidence(value) {
  return [
    "insufficient_evidence",
    "insufficient",
    "not_supported",
    "unsupported",
    "no_evidence",
    "no_relevant_evidence",
    "inadequate_evidence"
  ].includes(normalizeStatus(value));
}

function isPartialStatus(value) {
  return [
    "partially_supported",
    "partially_complete",
    "partial",
    "partially_sufficient"
  ].includes(normalizeStatus(value));
}

function Status({ value }) {
  const status = normalizeStatus(value);

  return (
    <span className={`status status-${status.replaceAll("_", "-")}`}>
      {status.replaceAll("_", " ")}
    </span>
  );
}

// --------------------------------------------------
// ANSWER FORMATTING
// --------------------------------------------------

function formatAnswer(answer) {
  if (!answer) return "";

  let text = String(answer);

  // Convert escaped Markdown symbols into normal Markdown.
  text = text.replace(/\\([#*_`~>-])/g, "$1");

  // Remove duplicate heading if it is already shown in the UI.
  text = text.replace(
    /^\s*#{1,3}\s*Research answer\s*/i,
    ""
  );

  // Remove a status word at the beginning of the answer.
  text = text.replace(
    /^\s*(supported|unsupported|partially supported|insufficient evidence)\s*/i,
    ""
  );

  return text.trim();
}

function removeUnsupportedCitations(answer) {
  if (!answer) return "";

  return String(answer)
    .replace(/\[\s*SOURCE\b[^\]]*\]/gi, "")
    .replace(/Supporting sources?:\s*SOURCE[\d\s,;]*/gi, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// --------------------------------------------------
// FORMATTED RESEARCH ANSWER
// --------------------------------------------------

function ResearchAnswer({ answer }) {
  if (!answer) return null;

  return (
    <div className="answer-body answer-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h2 className="markdown-heading">{children}</h2>
          ),

          h2: ({ children }) => (
            <h3 className="markdown-heading">{children}</h3>
          ),

          h3: ({ children }) => (
            <h4 className="markdown-subheading">{children}</h4>
          ),

          p: ({ children }) => (
            <p className="markdown-paragraph">{children}</p>
          ),

          strong: ({ children }) => (
            <strong className="markdown-strong">{children}</strong>
          ),

          ul: ({ children }) => (
            <ul className="markdown-list">{children}</ul>
          ),

          ol: ({ children }) => (
            <ol className="markdown-list">{children}</ol>
          ),

          li: ({ children }) => (
            <li className="markdown-list-item">{children}</li>
          ),

          blockquote: ({ children }) => (
            <blockquote className="markdown-quote">
              {children}
            </blockquote>
          ),

          code: ({ children }) => (
            <code className="markdown-code">{children}</code>
          ),

          hr: () => <hr className="markdown-divider" />
        }}
      >
        {answer}
      </ReactMarkdown>
    </div>
  );
}

// --------------------------------------------------
// DOCUMENT LIST
// --------------------------------------------------

function DocumentList({ documents, selectedIds, onToggle, loading }) {
  if (loading && !documents.length) {
    return <p className="empty">Loading papers…</p>;
  }

  if (!documents.length) {
    return (
      <div className="empty">
        <b>No papers uploaded yet</b>
        <span>Upload a PDF to start your research library.</span>
      </div>
    );
  }

  return (
    <div className="document-list">
      {documents.map((doc) => {
        const selected = selectedIds.includes(doc.document_id);

        return (
          <label
            className={`document-row ${selected ? "selected" : ""}`}
            key={doc.document_id}
          >
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle(doc.document_id)}
            />

            <span className="pdf-mark">PDF</span>

            <span className="doc-info">
              <b title={doc.filename}>{doc.filename}</b>
              <small>
                {doc.page_count ?? doc.pages ?? "—"} pages ·{" "}
                {doc.total_chunks ?? doc.chunks_count ?? "—"} chunks
              </small>
            </span>

            <span className="check-mark">
              {selected ? "✓" : ""}
            </span>
          </label>
        );
      })}
    </div>
  );
}

// --------------------------------------------------
// CLAIM VERIFICATION
// --------------------------------------------------

function ClaimList({ claims, forceUnsupported }) {
  if (!claims || claims.length === 0) {
    return (
      <div className="claims">
        <h3>Claim verification</h3>
        <p className="empty">
          No individual factual claims were returned for verification.
        </p>
      </div>
    );
  }

  return (
    <div className="claims">
      <h3>Claim verification</h3>

      {claims.map((claim, index) => {
        const claimStatus = forceUnsupported
          ? "unsupported"
          : claim.status || claim.verification_status;

        return (
          <div className="claim" key={claim.claim || index}>
            <Status value={claimStatus} />

            <p>
              {claim.claim ||
                claim.text ||
                claim.statement ||
                `Claim ${index + 1}`}
            </p>

            {claim.explanation && (
              <small>{claim.explanation}</small>
            )}

            {!forceUnsupported &&
              Array.isArray(claim.supporting_sources) &&
              claim.supporting_sources.length > 0 && (
                <small>
                  Supporting sources:{" "}
                  {claim.supporting_sources.join(", ")}
                </small>
              )}
          </div>
        );
      })}
    </div>
  );
}

// --------------------------------------------------
// ANSWER COMPLETENESS
// --------------------------------------------------

function CompletenessPanel({
  completeness,
  forceInsufficient,
  forcePartial
}) {
  if (!completeness || Object.keys(completeness).length === 0) {
    return null;
  }

  const missingFindings = Array.isArray(
    completeness.missing_findings
  )
    ? completeness.missing_findings
    : [];

  const status = forceInsufficient
    ? "insufficient_evidence"
    : forcePartial
    ? "partially_complete"
    : completeness.status;

  const explanation = forceInsufficient
    ? "The retrieved passages do not contain sufficient relevant evidence to answer the research question."
    : forcePartial
    ? "The answer covers the supported findings, but some parts of the research question remain unanswered because the retrieved evidence is incomplete."
    : completeness.explanation;

  return (
    <div className="completeness-panel">
      <div className="panel-head">
        <div>
          <h3>Answer completeness</h3>
          <p>
            Checks whether the answer addresses the important parts
            of the research question.
          </p>
        </div>

        <Status value={status} />
      </div>

      {explanation && <p>{explanation}</p>}

      {!forceInsufficient && missingFindings.length > 0 && (
        <div>
          <b>Missing findings</b>

          {missingFindings.map((finding, index) => (
            <div className="claim" key={index}>
              <p>{finding.finding}</p>

              {finding.explanation && (
                <small>{finding.explanation}</small>
              )}

              {finding.supporting_sources?.length > 0 && (
                <small>
                  Sources: {finding.supporting_sources.join(", ")}
                </small>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------
// RESEARCH COMPLETENESS
// --------------------------------------------------

function ResearchCompletenessPanel({ researchCompleteness }) {
  if (
    !researchCompleteness ||
    Object.keys(researchCompleteness).length === 0
  ) {
    return null;
  }

  const relevantSources = Array.isArray(
    researchCompleteness.relevant_sources
  )
    ? researchCompleteness.relevant_sources
    : [];

  const coverageGaps = Array.isArray(
    researchCompleteness.coverage_gaps
  )
    ? researchCompleteness.coverage_gaps
    : [];

  return (
    <div className="completeness-panel research-completeness-panel">
      <div className="panel-head">
        <div>
          <h3>Research completeness</h3>
          <p>
            Checks whether the retrieved passages contain relevant
            evidence for the research question.
          </p>
        </div>

        <Status value={researchCompleteness.status} />
      </div>

      {researchCompleteness.explanation && (
        <p>{researchCompleteness.explanation}</p>
      )}

      <div className="research-source-info">
        <b>Relevant sources</b>

        {relevantSources.length > 0 ? (
          <p>{relevantSources.join(", ")}</p>
        ) : (
          <p>No relevant supporting sources were identified.</p>
        )}
      </div>

      {coverageGaps.length > 0 && (
        <div className="research-coverage-gaps">
          <b>Missing research information</b>

          <ul>
            {coverageGaps.map((gap, index) => (
              <li key={index}>
                {typeof gap === "string"
                  ? gap
                  : gap.description || gap.gap || JSON.stringify(gap)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------
// RETRIEVED EVIDENCE
// --------------------------------------------------

function EvidenceList({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="evidence-list">
        <p className="empty">
          No retrieved passages are available.
        </p>
      </div>
    );
  }

  return (
    <div className="evidence-list">
      {results.map((item, index) => (
        <article
          className="evidence"
          key={`${item.document_id || item.filename}-${item.chunk_index ?? index}`}
        >
          <div className="evidence-meta">
            <b>SOURCE {item.rank || index + 1}</b>
            <strong>{item.filename || "Research paper"}</strong>
            <span>Chunk {item.chunk_index ?? "—"}</span>
          </div>

          <p>
            {item.text ||
              item.chunk ||
              item.chunk_text ||
              "No passage text returned."}
          </p>

          <small>
            {item.rerank_score != null
              ? `Reranker score: ${Number(item.rerank_score).toFixed(4)} · `
              : item.reranker_score != null
              ? `Reranker score: ${Number(item.reranker_score).toFixed(4)} · `
              : ""}

            {item.rrf_score != null
              ? `RRF score: ${Number(item.rrf_score).toFixed(4)}`
              : ""}
          </small>
        </article>
      ))}
    </div>
  );
}

// ==================================================
// MAIN APPLICATION
// ==================================================

export default function App() {
  const [theme, setTheme] = useState("light");
  const [page, setPage] = useState("workspace");

  const [documents, setDocuments] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);

  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);

  const [question, setQuestion] = useState("");
  const [answerResult, setAnswerResult] = useState(null);

  const [error, setError] = useState("");

  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const fileRef = useRef(null);

  const selectedCount = selectedIds.length;

  const allSelected =
    documents.length > 0 &&
    documents.every((doc) =>
      selectedIds.includes(doc.document_id)
    );

  // --------------------------------------------------
  // VERIFICATION DATA
  // --------------------------------------------------

  const verification = answerResult?.verification || {};

  const researchCompleteness =
    verification.research_completeness || {};

  const researchStatus = researchCompleteness.status;

  const forceInsufficient =
    isInsufficientEvidence(researchStatus) ||
    (
      Array.isArray(researchCompleteness.relevant_sources) &&
      researchCompleteness.relevant_sources.length === 0 &&
      normalizeStatus(researchStatus) !== "supported" &&
      Boolean(researchStatus)
    );

  const claims =
    verification.claims ||
    verification.claim_results ||
    [];

  const completeness = verification.completeness || {};

  const results =
    answerResult?.retrieved_results ||
    answerResult?.results ||
    [];

  const overallStatus = forceInsufficient
    ? "unsupported"
    : verification.overall_status || verification.status;

  const forcePartial =
    !forceInsufficient &&
    (
      isPartialStatus(overallStatus) ||
      isPartialStatus(researchStatus) ||
      researchCompleteness.coverage_gaps?.length > 0
    );

  const rawAnswer = answerResult?.answer || "";

  const cleanedAnswer = formatAnswer(rawAnswer);

  const displayedAnswer = forceInsufficient
    ? removeUnsupportedCitations(cleanedAnswer)
    : cleanedAnswer;

  // --------------------------------------------------
  // LOAD DOCUMENTS
  // --------------------------------------------------

  const fetchDocuments = async () => {
    setLoadingDocs(true);

    try {
      const response = await axios.get(
        `${API_URL}/api/documents`
      );

      const list = response.data?.documents || [];

      setDocuments(list);

      setSelectedIds((oldIds) =>
        oldIds.filter((id) =>
          list.some((doc) => doc.document_id === id)
        )
      );
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Cannot connect to Flask. Check that the backend is running."
      );
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // --------------------------------------------------
  // SELECT OR DESELECT DOCUMENT
  // --------------------------------------------------

  const toggleDocument = (id) => {
    setSelectedIds((oldIds) =>
      oldIds.includes(id)
        ? oldIds.filter((item) => item !== id)
        : [...oldIds, id]
    );

    setAnswerResult(null);
  };

  // --------------------------------------------------
  // UPLOAD PDF
  // --------------------------------------------------

  const handleUpload = async (event) => {
    event.preventDefault();

    setError("");
    setUploadResult(null);

    if (!file) {
      setError("Please select a PDF file.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    const form = new FormData();
    form.append("file", file);

    setUploading(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/upload`,
        form,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          },
          timeout: 180000
        }
      );

      const data = response.data || {};
      const doc = data.document || data;

      setUploadResult({
        ...data,
        filename: data.filename || doc.filename || file.name,

        page_count:
          data.page_count ??
          doc.page_count ??
          doc.pages,

        total_chunks:
          data.total_chunks ??
          data.chunks_created ??
          doc.total_chunks ??
          doc.chunks_created,

        text_preview:
          data.text_preview ||
          data.extracted_preview ||
          "",

        chunks_preview:
          data.chunks_preview ||
          data.chunks ||
          []
      });

      if (doc.document_id) {
        setSelectedIds((oldIds) =>
          oldIds.includes(doc.document_id)
            ? oldIds
            : [...oldIds, doc.document_id]
        );
      }

      setFile(null);

      if (fileRef.current) {
        fileRef.current.value = "";
      }

      await fetchDocuments();
    } catch (err) {
      setError(
        err.response?.data?.error ||
          err.response?.data?.message ||
          err.message ||
          "PDF upload failed."
      );
    } finally {
      setUploading(false);
    }
  };

  // --------------------------------------------------
  // ASK RESEARCH QUESTION
  // --------------------------------------------------

  const handleAsk = async (event) => {
    event.preventDefault();

    setError("");
    setAnswerResult(null);

    if (!question.trim()) {
      setError("Enter a research question.");
      return;
    }

    if (!selectedCount) {
      setError("Select at least one research paper.");
      return;
    }

    setAsking(true);

    try {
      const response = await axios.post(
        `${API_URL}/api/ask`,
        {
          question: question.trim(),
          selected_document_ids: selectedIds
        },
        {
          timeout: 180000
        }
      );

      const data = response.data || {};

      const retrieved =
        data.retrieved_results ||
        data.results ||
        [];

      setAnswerResult({
        ...data,

        answer:
          data.answer ||
          "No answer was returned.",

        verification:
          data.verification ||
          {},

        retrieved_results: retrieved,

        retrieved_count:
          data.retrieved_count ??
          data.retrieved_chunks ??
          retrieved.length
      });
    } catch (err) {
      setError(
        err.response?.data?.error ||
          err.response?.data?.message ||
          err.message ||
          "Could not generate an answer."
      );
    } finally {
      setAsking(false);
    }
  };

  // ==================================================
  // MAIN UI
  // ==================================================

  return (
    <div className={`app-shell ${theme}`}>

      {/* SIDEBAR */}
      <aside className="sidebar">

        <div className="brand">
          <span className="brand-logo">R</span>

          <span>
            <b>ResearchAI</b>
            <small>Scientific workspace</small>
          </span>
        </div>

        <div className="side-label">WORKSPACE</div>

        <button
          className={`nav-item ${page === "workspace" ? "active" : ""}`}
          onClick={() => setPage("workspace")}
        >
          <i>⌂</i>
          Research workspace
        </button>

        <button
          className={`nav-item ${page === "papers" ? "active" : ""}`}
          onClick={() => setPage("papers")}
        >
          <i>▤</i>
          My research papers
          <em>{documents.length}</em>
        </button>

        <div className="side-label dev-label">DEVELOPER</div>

        <button
          className={`nav-item ${page === "developer" ? "active" : ""}`}
          onClick={() => setPage("developer")}
        >
          <i>▥</i>
          Developer analytics
        </button>

        <div className="side-bottom">
          <div className="side-tip">
            <span>✦</span>
            <b>Research assistant</b>
            <p>
              Ask questions grounded in your selected scientific papers.
            </p>
          </div>

          <small>Agentic RAG · MCA Project</small>
        </div>

      </aside>

      {/* MAIN AREA */}
      <div className="main-area">

        {/* TOP BAR */}
        <header className="topbar">

          <div className="crumb">
            ResearchAI
            <span>/</span>
            <b>
              {page === "developer"
                ? "Developer analytics"
                : page === "papers"
                ? "My research papers"
                : "Research workspace"}
            </b>
          </div>

          <div className="top-actions">

            <span className="api-state">
              <i />
              API workspace
            </span>

            <button
              className="theme-btn"
              onClick={() =>
                setTheme((currentTheme) =>
                  currentTheme === "light" ? "dark" : "light"
                )
              }
              aria-label="Toggle theme"
            >
              {theme === "light" ? "☾" : "☀"}
            </button>

            <span className="avatar">RA</span>

          </div>

        </header>

        {/* PAGE CONTENT */}
        <main className="content">

          {/* DEVELOPER DASHBOARD */}
          {page === "developer" ? (

            <DeveloperDashboard />

          ) : page === "papers" ? (

            /* DOCUMENT LIBRARY */
            <section>

              <div className="page-head">
                <div>
                  <label>DOCUMENT LIBRARY</label>
                  <h1>My research papers</h1>
                  <p>
                    Manage the scientific PDFs indexed in this session.
                  </p>
                </div>

                <button
                  className="primary"
                  onClick={() => setPage("workspace")}
                >
                  + Upload a paper
                </button>
              </div>

              <div className="panel">

                <div className="panel-head">
                  <div>
                    <h2>Research library</h2>
                    <p>{documents.length} paper(s) available</p>
                  </div>

                  <button className="link-btn" onClick={fetchDocuments}>
                    ↻ Refresh
                  </button>
                </div>

                <DocumentList
                  documents={documents}
                  selectedIds={selectedIds}
                  onToggle={toggleDocument}
                  loading={loadingDocs}
                />

              </div>

            </section>

          ) : (

            /* RESEARCH WORKSPACE */
            <section>

              <div className="page-head">

                <div>
                  <label>SCIENTIFIC KNOWLEDGE ASSISTANT</label>
                  <h1>Research workspace</h1>
                  <p>
                    Upload papers, ask questions, and inspect the supporting evidence.
                  </p>
                </div>

                <span className="ready-chip">
                  <i />
                  Ready for research
                </span>

              </div>

              {error && (
                <div className="error-banner">
                  <b>!</b>
                  <span>{error}</span>
                  <button onClick={() => setError("")}>×</button>
                </div>
              )}

              <div className="workspace">

                {/* LEFT SIDE */}
                <div className="left-stack">

                  {/* UPLOAD PANEL */}
                  <div className="panel">

                    <div className="panel-head">
                      <div className="head-icon">↑</div>

                      <div>
                        <h2>Upload research paper</h2>
                        <p>Add a PDF to your searchable library.</p>
                      </div>
                    </div>

                    <form onSubmit={handleUpload}>

                      <label className="dropzone">

                        <input
                          ref={fileRef}
                          type="file"
                          accept=".pdf,application/pdf"
                          onChange={(event) => {
                            setFile(event.target.files?.[0] || null);
                            setUploadResult(null);
                            setError("");
                          }}
                        />

                        <span className="upload-icon">⇧</span>

                        <b>
                          {file ? file.name : "Choose a PDF to upload"}
                        </b>

                        <small>
                          {file ? "Selected file" : "Click to browse · PDF format"}
                        </small>

                      </label>

                      <button className="primary wide" disabled={uploading}>
                        {uploading
                          ? "Uploading and indexing…"
                          : "Upload and process PDF"}
                      </button>

                    </form>

                    {uploadResult && (
                      <div className="upload-success">

                        <b>✓ Paper processed successfully</b>

                        <div className="upload-stats">

                          <span>
                            <strong>{uploadResult.page_count ?? "—"}</strong>
                            Pages
                          </span>

                          <span>
                            <strong>{uploadResult.total_chunks ?? "—"}</strong>
                            Chunks
                          </span>

                          <span>
                            <strong>{uploadResult.embedding_dimension ?? 384}</strong>
                            Dimensions
                          </span>

                        </div>

                        <small>{uploadResult.filename}</small>

                      </div>
                    )}

                  </div>

                  {/* RESEARCH LIBRARY */}
                  <div className="panel">

                    <div className="panel-head">

                      <div>
                        <h2>Research library</h2>
                        <p>
                          {documents.length} paper(s) · {selectedCount} selected
                        </p>
                      </div>

                      <button className="link-btn" onClick={fetchDocuments}>
                        ↻ Refresh
                      </button>

                    </div>

                    <div className="select-all">

                      <label>
                        <input
                          type="checkbox"
                          checked={allSelected}
                          onChange={() => {
                            setSelectedIds(
                              allSelected
                                ? []
                                : documents.map((doc) => doc.document_id)
                            );

                            setAnswerResult(null);
                          }}
                        />

                        Select all papers
                      </label>

                      <span>{selectedCount} selected</span>

                    </div>

                    <DocumentList
                      documents={documents}
                      selectedIds={selectedIds}
                      onToggle={toggleDocument}
                      loading={loadingDocs}
                    />

                  </div>

                </div>

                {/* RIGHT SIDE */}
                <div className="right-stack">

                  {/* ASK QUESTION PANEL */}
                  <div className="panel chat-panel">

                    <div className="chat-top">

                      <span className="ai-mark">✦</span>

                      <div>
                        <h2>Ask ResearchAI</h2>
                        <p>
                          Answers grounded in retrieved paper evidence
                        </p>
                      </div>

                      <i className="online-dot" />

                    </div>

                    <div className="welcome">

                      <span>✧</span>
                      <h3>What would you like to explore?</h3>

                      <p>
                        Ask a question about your selected papers.
                        ResearchAI retrieves relevant passages, reranks them,
                        generates an answer, and checks its evidence.
                      </p>

                    </div>

                    <form className="ask-form" onSubmit={handleAsk}>

                      <label htmlFor="question">
                        Your research question
                      </label>

                      <textarea
                        id="question"
                        rows="4"
                        value={question}
                        onChange={(event) => setQuestion(event.target.value)}
                        placeholder="Example: What is Agentic RAG and how does it differ from traditional RAG?"
                      />

                      <div className="ask-footer">

                        <span>{selectedCount} paper(s) selected</span>

                        <button className="primary" disabled={asking}>
                          {asking ? "Researching…" : "Generate answer →"}
                        </button>

                      </div>

                    </form>

                    {asking && (
                      <div className="loading-note">
                        Searching, reranking, and verifying evidence…
                      </div>
                    )}

                  </div>

                  {/* ANSWER PANEL */}
                  {answerResult && (

                    <div className="panel answer-panel">

                      <div className="answer-heading">

                        <div>
                          <label>GENERATED RESPONSE</label>
                          <h2>Research answer</h2>
                        </div>

                        <Status value={overallStatus} />

                      </div>

                      {/* FORMATTED ANSWER */}
                      <ResearchAnswer answer={displayedAnswer} />

                      {/* PARTIAL EVIDENCE NOTICE */}
                      {forcePartial && (
                        <div className="missing-evidence-notice">
                          <h3>Some research information is missing</h3>
                          <p>
                            The retrieved papers support only part of this
                            question. The unsupported part is identified
                            below rather than being presented as a confirmed
                            research finding.
                          </p>
                        </div>
                      )}

                      {/* VERIFICATION SUMMARY */}
                      {verification.summary && (
                        <p className="verify-summary">
                          {forceInsufficient
                            ? "The retrieved research passages do not provide relevant evidence. The answer is therefore not supported by the selected papers."
                            : verification.summary}
                        </p>
                      )}

                      {/* CLAIM VERIFICATION */}
                      <ClaimList
                        claims={claims}
                        forceUnsupported={forceInsufficient}
                      />

                      {/* ANSWER COMPLETENESS */}
                      <CompletenessPanel
                        completeness={completeness}
                        forceInsufficient={forceInsufficient}
                        forcePartial={forcePartial}
                      />

                      {/* RESEARCH COMPLETENESS */}
                      <ResearchCompletenessPanel
                        researchCompleteness={researchCompleteness}
                      />

                      {/* RETRIEVED EVIDENCE */}
                      <div className="evidence-head">

                        <div>
                          <h3>Retrieved evidence</h3>
                          <p>
                            {answerResult.retrieved_count ?? results.length}
                            {" "}passage(s) returned
                          </p>
                        </div>

                      </div>

                      <EvidenceList results={results} />

                    </div>

                  )}

                </div>

              </div>

            </section>

          )}

        </main>

      </div>

    </div>
  );
}
