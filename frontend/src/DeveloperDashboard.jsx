
import React, { useMemo, useState } from "react";
import "./DeveloperDashboard.css";

const rows = [
  { method: "Vector Search", recall: .7883, precision: .0890, mrr: .6068 },
  { method: "BM25 Search", recall: .6928, precision: .0770, mrr: .5326 },
  { method: "Hybrid Search", recall: .7891, precision: .0877, mrr: .6214 },
  { method: "Hybrid + CrossEncoder", recall: .8212, precision: .0927, mrr: .6629 },
  { method: "Hybrid + MedCPT", recall: .8629, precision: .0973, mrr: .7484 }
];

const metrics = [
  { key: "recall", title: "Recall@10", desc: "Relevant documents found in the top 10.", meaning: "Recall measures how many of the relevant documents were successfully retrieved. Higher recall means fewer relevant documents are missed." },
  { key: "precision", title: "Precision@10", desc: "Relevant results among the top 10.", meaning: "Precision measures how many of the retrieved top 10 documents are relevant. Higher precision means fewer irrelevant documents appear." },
  { key: "mrr", title: "MRR@10", desc: "Ranking position of the first relevant result.", meaning: "Mean Reciprocal Rank rewards systems that place the first relevant document near the top of the result list." }
];

const components = [
  { id: "01", name: "Embedding model", technology: "all-MiniLM-L6-v2", description: "Converts text into 384-dimensional vectors.", input: "Research paper text and user query", output: "384-dimensional numerical vectors", importance: "Enables semantic similarity matching, even when the query and paper use different words." },
  { id: "02", name: "Vector index", technology: "FAISS", description: "Finds semantically similar passages.", input: "Query embedding and document embeddings", output: "Ranked semantically similar passages", importance: "Provides fast similarity-based retrieval from the scientific document collection." },
  { id: "03", name: "Keyword retrieval", technology: "BM25", description: "Ranks passages using keyword matching.", input: "User query and indexed document text", output: "Keyword-ranked passages", importance: "Finds documents containing important terms and exact scientific keywords." },
  { id: "04", name: "Fusion strategy", technology: "Reciprocal Rank Fusion", description: "Combines vector and BM25 rankings.", input: "Ranked results from vector and BM25 search", output: "A combined hybrid ranking", importance: "Uses both semantic and keyword retrieval to improve candidate selection." },
  { id: "05", name: "Reranker", technology: "MedCPT Cross-Encoder", description: "Reorders scientific passages by relevance.", input: "User query and hybrid candidate passages", output: "Relevance-ranked passages", importance: "Provides more detailed query-passage relevance scoring before answer generation." },
  { id: "06", name: "Answer generation", technology: "Gemini", description: "Generates an answer from retrieved evidence.", input: "User question and selected evidence passages", output: "Natural-language research answer", importance: "Produces a readable answer grounded in the retrieved research evidence." },
  { id: "07", name: "Evidence verification", technology: "LLM Claim Verifier", description: "Checks answer claims against evidence.", input: "Generated answer and retrieved passages", output: "Evidence support assessment", importance: "Helps identify claims that are supported or unsupported by the retrieved passages." }
];

const steps = [
  ["01", "User query", "Research question submitted."],
  ["02", "Hybrid retrieval", "Vector + BM25 rankings fused using RRF."],
  ["03", "MedCPT reranking", "Candidate passages reordered by relevance."],
  ["04", "Gemini generation", "Answer generated from retrieved passages."],
  ["05", "Evidence verification", "Claims checked against retrieved evidence."]
];

function MetricBars({ metric, visibleRows }) {
  const info = metrics.find(m => m.key === metric);
  const max = Math.max(0, ...visibleRows.map(r => r[metric]));

  return (
    <div className="dev-chart panel">
      <div className="chart-title">
        <div>
          <h3>{info.title}</h3>
          <p>{info.desc}</p>
        </div>
        <span>TOP 10</span>
      </div>

      {visibleRows.length === 0 ? (
        <p className="interactive-empty">Select at least one method to display the chart.</p>
      ) : (
        visibleRows.map(r => (
          <div className="bar-row" key={r.method}>
            <div>
              <span>{r.method}</span>
              <b>{r[metric].toFixed(4)}</b>
            </div>
            <div className="bar-track">
              <i
                className={r[metric] === max ? "best" : ""}
                style={{ width: `${r[metric] * 100}%` }}
              />
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function downloadCSV(data) {
  if (!data.length) return;

  const header = ["Method", "Recall@10", "Precision@10", "MRR@10"];
  const csvRows = data.map(r => [
    r.method,
    r.recall.toFixed(4),
    r.precision.toFixed(4),
    r.mrr.toFixed(4)
  ]);

  const csv = [header, ...csvRows]
    .map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  const blob = new Blob(["\uFEFF" + csv], {
    type: "text/csv;charset=utf-8;"
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "ResearchAI_Evaluation_Results.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function DeveloperDashboard() {
  const initialMethods = rows.map(r => r.method);

  const [selectedMetric, setSelectedMetric] = useState("recall");
  const [selectedMethods, setSelectedMethods] = useState(initialMethods);
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedComponent, setExpandedComponent] = useState(null);

  const visibleRows = useMemo(() => {
    return rows.filter(r =>
      selectedMethods.includes(r.method) &&
      r.method.toLowerCase().includes(searchTerm.toLowerCase().trim())
    );
  }, [selectedMethods, searchTerm]);

  const best = rows.reduce((a, b) => b.mrr > a.mrr ? b : a);

  const selectedMetricInfo = metrics.find(m => m.key === selectedMetric);

  const toggleMethod = method => {
    setSelectedMethods(current =>
      current.includes(method)
        ? current.filter(m => m !== method)
        : [...current, method]
    );
  };

  return (
    <div className="developer">
      <div className="page-head">
        <div>
          <label>DEVELOPER / SYSTEM ANALYSIS</label>
          <h1>Developer analytics</h1>
          <p>Explore the dataset, technical pipeline, and saved retrieval experiments.</p>
        </div>
        <span className="saved-chip"><i /> Saved Colab results</span>
      </div>

      <div className="overview-strip">
        {[
          ["DATASET", "SciFact", "Scientific claim verification"],
          ["EMBEDDINGS", "MiniLM-L6-v2", "384 dimensions"],
          ["RETRIEVAL", "Hybrid + RRF", "Vector + BM25"],
          ["RERANKER", "MedCPT", "Cross-encoder"]
        ].map(x => (
          <div key={x[0]}>
            <small>{x[0]}</small>
            <b>{x[1]}</b>
            <span>{x[2]}</span>
          </div>
        ))}
      </div>

      <section className="dev-section">
        <div className="section-title">
          <label>01 / DATA</label>
          <h2>Dataset overview</h2>
          <p>Confirm these figures against the final dataset split loaded in Colab.</p>
        </div>

        <div className="dev-metrics">
          {[
            ["Corpus documents", "5,183", "Scientific paper records"],
            ["Evaluation queries", "300", "Research questions"],
            ["Qrels", "300", "Relevance judgments · verify count"],
            ["Embedding dimensions", "384", "all-MiniLM-L6-v2"]
          ].map(x => (
            <div className="metric-tile" key={x[0]}>
              <span>{x[0]}</span>
              <b>{x[1]}</b>
              <small>{x[2]}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="dev-section">
        <div className="section-title">
          <label>02 / ARCHITECTURE</label>
          <h2>System components</h2>
          <p>Click a component to understand its input, output, and importance.</p>
        </div>

        <div className="component-grid">
          {components.map(x => (
            <article
              className={`component-tile component-interactive ${expandedComponent === x.id ? "component-open" : ""}`}
              key={x.id}
            >
              <span className="component-number">{x.id}</span>
              <div>
                <button
                  type="button"
                  className="component-toggle"
                  onClick={() => setExpandedComponent(
                    expandedComponent === x.id ? null : x.id
                  )}
                  aria-expanded={expandedComponent === x.id}
                >
                  <small>{x.name}</small>
                  <b>{x.technology}</b>
                  <span className="component-hint">
                    {expandedComponent === x.id ? "− Hide details" : "+ View details"}
                  </span>
                </button>
                <p>{x.description}</p>

                {expandedComponent === x.id && (
                  <div className="component-details">
                    <p><strong>Input:</strong> {x.input}</p>
                    <p><strong>Output:</strong> {x.output}</p>
                    <p><strong>Importance:</strong> {x.importance}</p>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="dev-section">
        <div className="section-title">
          <label>03 / EXPERIMENTS</label>
          <h2>Retrieval performance</h2>
          <p>Filter, search, compare, and export the saved evaluation results.</p>
        </div>

        <div className="panel dev-controls">
          <div className="control-heading">
            <div>
              <h3>Explore evaluation results</h3>
              <p>Choose a metric and the retrieval methods you want to compare.</p>
            </div>
            <button
              type="button"
              className="export-button"
              onClick={() => downloadCSV(visibleRows)}
              disabled={visibleRows.length === 0}
            >
              ↓ Export CSV
            </button>
          </div>

          <div className="control-group">
            <label htmlFor="metric-select">Performance metric</label>
            <select
              id="metric-select"
              value={selectedMetric}
              onChange={e => setSelectedMetric(e.target.value)}
            >
              {metrics.map(m => (
                <option value={m.key} key={m.key}>{m.title}</option>
              ))}
            </select>
          </div>

          <div className="method-heading">
            <label>Retrieval methods</label>
            <div>
              <button type="button" onClick={() => setSelectedMethods(initialMethods)}>
                Select all
              </button>
              <button type="button" onClick={() => setSelectedMethods([])}>
                Clear all
              </button>
            </div>
          </div>

          <div className="method-filters">
            {rows.map(r => (
              <label className="method-option" key={r.method}>
                <input
                  type="checkbox"
                  checked={selectedMethods.includes(r.method)}
                  onChange={() => toggleMethod(r.method)}
                />
                <span>{r.method}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="panel table-panel">
          <div className="table-heading">
            <div>
              <h3>Evaluation comparison</h3>
              <p>Showing {visibleRows.length} of {rows.length} retrieval methods.</p>
            </div>
            <span>COLAB EXPERIMENT</span>
          </div>

          <div className="table-search">
            <input
              type="search"
              placeholder="Search retrieval method..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              aria-label="Search retrieval methods"
            />
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Recall@10</th>
                  <th>Precision@10</th>
                  <th>MRR@10</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map(r => (
                  <tr className={r.method === best.method ? "best-row" : ""} key={r.method}>
                    <td>
                      <b>{r.method}</b>
                      {r.method === best.method && <em>Best</em>}
                    </td>
                    <td>{r.recall.toFixed(4)}</td>
                    <td>{r.precision.toFixed(4)}</td>
                    <td>{r.mrr.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {visibleRows.length === 0 && (
              <p className="empty-table">No methods match your selection.</p>
            )}
          </div>
        </div>

        <div className="metric-explanation">
          <div>
            <strong>{selectedMetricInfo.title}</strong>
            <p>{selectedMetricInfo.meaning}</p>
          </div>
          <span>Selected metric</span>
        </div>

        <div className="charts-grid">
          <MetricBars metric={selectedMetric} visibleRows={visibleRows} />
        </div>
      </section>

      <section className="dev-section">
        <div className="section-title">
          <label>04 / RESULT</label>
          <h2>Best-performing configuration</h2>
          <p>Highest scores among the evaluated retrieval methods.</p>
        </div>

        <div className="best-panel">
          <div>
            <small>BEST EVALUATED METHOD</small>
            <h3>Hybrid Search + MedCPT</h3>
            <p>Combines semantic and keyword retrieval, rank fusion, and scientific passage reranking.</p>
          </div>
          <div className="best-values">
            <div><b>{best.recall.toFixed(4)}</b><span>Recall@10</span></div>
            <div><b>{best.precision.toFixed(4)}</b><span>Precision@10</span></div>
            <div><b>{best.mrr.toFixed(4)}</b><span>MRR@10</span></div>
          </div>
        </div>
      </section>

      <section className="dev-section">
        <div className="section-title">
          <label>05 / PIPELINE</label>
          <h2>Research assistant workflow</h2>
          <p>How a question moves through the current application.</p>
        </div>

        <div className="workflow">
          {steps.map((s, i) => (
            <React.Fragment key={s[0]}>
              <div className="workflow-step">
                <span>{s[0]}</span>
                <b>{s[1]}</b>
                <p>{s[2]}</p>
              </div>
              {i < steps.length - 1 && (
                <span className="workflow-arrow">→</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </section>

      <div className="dev-note">
        <b>Experiment note:</b> These are saved Google Colab evaluation values.
        Dashboard filters and charts operate on these saved results; they do not
        rerun SciFact evaluation or calculate live metrics. Verify corpus, query,
        and qrels counts, and make the Gemini label match the model configured
        in your Flask backend.
      </div>
    </div>
  );
}
