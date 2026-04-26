# PreTrainAudit

**A content-neutral pre-training dataset risk quantification platform.**

> "Before you train, know your risks."

PreTrainAudit is a locally-runnable, SaaS-style analytical engine that audits datasets for structural safety, training risks, and AI model compatibility before a single GPU cycle is spent.

---

## What Makes It Novel

Current dataset safety tools are primarily **content-based** (they judge what the data says). 
PreTrainAudit is entirely **structure-based** (it measures how the data behaves statistically during training).

This represents a paradigm shift: dataset safety is not content moderation. Dataset safety is quantifiable training risk.

### Core Risk Modules

| Module | Purpose |
|--------|---------|
| **Adversarial Scanner** | Detects mislabeled clusters and poisoning-susceptible regions via embedding-space geometry. |
| **Memorization Scorer** | Estimates privacy leakage probability dynamically using token-to-parameter ratios and dataset duplication metrics. |
| **Fairness Drift** | Forecasts production fairness degradation risk using a Peak + Breadth Gini impurity model across categorical columns. |
| **Contamination Detector** | Identifies train/test leakage by cross-referencing multi-word statistical fingerprints against known public benchmarks. |
| **Cost Estimator** | Calculates required FLOPs and accurate GPU training hours (scaled to hardware capabilities) before training begins. |
| **Compatibility Advisor** | Detects architecture-dataset mismatches and provides actionable, data-driven mitigation strategies. |

---

## Architecture and Engineering

- **Backend Framework**: Built on FastAPI for high-performance, asynchronous API routing.
- **Out-of-Core Processing**: Leverages DuckDB and Dask to analyze massive datasets (MBs to TBs) using intelligent chunking without loading the full data into RAM.
- **Local-First Security**: Operates entirely on the user's hardware. Sensitive training data never leaves the host machine.
- **Statistical Rigor**: Utilizes auto-scaling stratified sampling and bootstrap-based confidence intervals to quantify uncertainty in risk scores.
- **Reproducibility**: Enforces deterministic random seeding across all stochastic operations (like adversarial sampling) for consistent auditing.
- **Resource Management**: Implements TTL-based session expiration with robust background janitor tasks to prevent memory leaks.
- **Frontend Integration**: Ships with a vanilla HTML/JS dashboard for seamless visualization without heavy Node.js dependencies.

---

## Quickstart Guide

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application engine:
   ```bash
   ../start.bat
   ```
   *(Alternatively, run `python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload`)*

4. Open your web browser and navigate to `http://localhost:8000`.

*Note: Do not open the frontend HTML files directly in your browser. Always access the application through the local web server to ensure API connectivity.*

---

## Supported Data Formats

- **Tabular**: CSV, Parquet, Excel.
- **Text**: TXT, JSONL, JSON, plain text corpora.
- **Image**: (Experimental) Folder collections of JPG/PNG for metadata and distribution analysis.

---

## Research Context

PreTrainAudit formalizes content-neutral pre-training dataset risk into a suite of measurable, architecture-aware metrics. It serves as the dataset-layer counterpart to established model-layer safety tools, complementing efforts in activation-space adversarial detection and temporal fairness monitoring.

---

## Security Considerations

This tool is designed explicitly for local-first execution. 
- Ensure your environment is secure when analyzing sensitive datasets.
- The built-in session manager automatically purges uploaded data from the temporary directories upon session expiration.
- For production deployment, you must tighten CORS policies and deploy behind a secure reverse proxy.

---

## Contributing

Contributions to PreTrainAudit are welcome. Areas of interest include adding new domain detectors, expanding benchmark contamination fingerprints, and optimizing out-of-core profiling algorithms.
