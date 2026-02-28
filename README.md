# InvoiceFlow AI

Automated invoice processing pipeline that uses local LLMs (via Ollama) for
OCR extraction, semantic vendor matching, AI-based fraud auditing, and 3-way
PO matching. A Streamlit dashboard provides batch upload and human review.

---

## Architecture

```
┌──────────────┐
│   app.py     │  Streamlit UI — file upload, batch trigger, result display
└──────┬───────┘
       │ calls
       ▼
┌──────────────┐     ┌────────────────┐
│ ocr_engine   │────▶│ Ollama API     │  POST /api/chat  (llama3.2-vision)
│  .py         │     │ localhost:11434 │  Base64 image → structured JSON
└──────┬───────┘     └────────────────┘
       │ returns dict
       ▼
┌──────────────┐
│ processor.py │  Pipeline orchestrator
│              │
│  1. Vendor   │──▶ vector_db.smart_match_vendor()
│     Match    │      embeddinggemma via Ollama /api/embeddings
│              │      cosine similarity, 0.6 threshold
│              │      fallback → rapidfuzz token_sort_ratio (80% threshold)
│              │
│  2. Validate │──▶ Rule-based: duplicates, missing fields, high-value (>$10k)
│              │
│  3. AI Audit │──▶ Ollama /api/chat (llama3) — forensic fraud scoring
│              │
│  4. 3-Way    │──▶ erp_mock.check_3_way_match()
│     Match    │      PO lookup, $1 tolerance variance check
└──────────────┘
```

### Module Dependency Graph

```
app.py
├── ocr_engine.py      (requests, base64, json)
└── processor.py        (pandas, rapidfuzz, requests, json)
    ├── vector_db.py    (requests, numpy)
    └── erp_mock.py     (no external deps)
```

---

## Modules

| File | Purpose | Ollama Model |
|------|---------|-------------|
| `app.py` | Streamlit dashboard — upload, batch process, review | — |
| `ocr_engine.py` | Vision-LLM extraction of invoice images to JSON | `llama3.2-vision` |
| `processor.py` | Vendor normalization, validation, AI audit, 3-way match | `llama3` |
| `vector_db.py` | Semantic vendor matching via embeddings + cosine similarity | `embeddinggemma` (fallback: `all-minilm`) |
| `erp_mock.py` | In-memory PO database and 3-way match logic | — |

---

## Execution Flow

1. **Upload** — User uploads PDF/image files via Streamlit sidebar.
2. **Temp file** — Each upload is written to a `tempfile.NamedTemporaryFile`; cleaned up in a `finally` block.
3. **OCR** — `ocr_engine.extract_invoice_data()` base64-encodes the image, POSTs to Ollama `llama3.2-vision`, strips markdown fences, and returns parsed JSON. On failure returns `{"error": "OCR processing failed", "file": ...}`.
4. **Error gate** — `app.py` checks for `"error"` key; if present, populates default fields and skips the pipeline.
5. **Vendor matching** — `processor.process_pipeline()` calls `vector_db.smart_match_vendor()` (embedding cosine similarity, threshold 0.6). If it returns "Unknown" or "New Vendor", falls back to `match_vendor()` (rapidfuzz `token_sort_ratio`, threshold 80%).
6. **Validation** — `validate_invoice()` checks: duplicate (via in-memory `PROCESSED_DB` DataFrame), high value (> $10,000 after `float()` coercion), missing invoice number.
7. **Duplicate tracking** — After validation, the invoice is appended to `PROCESSED_DB` via `pd.concat` so future calls detect duplicates.
8. **AI audit** — `audit_invoice_with_ai()` sends the invoice JSON to `llama3` with a forensic-accountant prompt; returns `{risk_score, flags}`. On failure returns `{"risk_score": 0, "flags": ["Audit unavailable"]}`.
9. **3-way match** — `erp_mock.check_3_way_match()` scans invoice keys for `po` + `number` substrings, looks up the PO in the `POS` dict, and compares totals with a $1 tolerance.
10. **Display** — Results are shown in a `pd.json_normalize` DataFrame with per-invoice drill-down for extracted data, validation flags, PO match status, and AI audit score.

---

## Mock Data

### Vendor Master (`vector_db.VENDORS`)

Amazon Web Services, Microsoft Azure, Google Cloud Platform, Staples Office Supplies, WeWork Space, Delta Airlines, Uber Business

### Fuzzy Vendor List (`processor.KNOWN_VENDORS`)

Amazon Web Services, Microsoft Azure, GitHub Inc, Staples, WeWork

### Purchase Orders (`erp_mock.POS`)

| PO | Vendor | Approved Total |
|----|--------|---------------|
| PO-1001 | Amazon Web Services | $5,000.00 |
| PO-1002 | Staples | $250.50 |

---

## Setup

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally (`ollama serve`)

### Installation

```bash
git clone https://github.com/pypi-ahmad/invoiceflow-ai.git
cd invoiceflow-ai

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pull Ollama Models

All three models must be pulled before running the application:

```bash
ollama pull llama3.2-vision   # OCR extraction (ocr_engine.py)
ollama pull llama3             # Fraud audit    (processor.py)
ollama pull embeddinggemma     # Vendor match   (vector_db.py)
```

If `embeddinggemma` is unavailable, `vector_db.py` falls back to `all-minilm`:

```bash
ollama pull all-minilm         # Optional fallback embedding model
```

---

## Dependencies

From `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard UI |
| `pandas` | DataFrames for result display and duplicate tracking |
| `rapidfuzz` | Fuzzy string matching (vendor fallback) |
| `requests` | HTTP calls to Ollama REST API |
| `numpy` | Cosine similarity computation |
| `watchdog` | Listed but unused in current code |
| `plotly` | Listed but unused in current code |

### Test Dependencies (dev)

```bash
pip install pytest pytest-cov pytest-mock
```

---

## Running the Application

```bash
cd invoice-pipeline
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Upload invoice images/PDFs via
the sidebar, click **Run Batch Process**, and review results in the main panel.

---

## Running Tests

```bash
cd invoice-pipeline

# Standard run
python -m pytest tests/ -v --tb=short

# Strict mode (RuntimeWarnings → errors)
python -m pytest tests/ -v --tb=short -W error::RuntimeWarning

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Testing

- **108 tests** — 0 failures, 0 skipped, 0 warnings
- **92% overall coverage** — all business-logic modules ≥ 96%
- **Fully deterministic** — no external dependencies; all LLM/network calls mocked
- **State-isolated** — autouse fixture resets `PROCESSED_DB` before every test

### Test Suite Breakdown

| File | Tests | Scope |
|------|-------|-------|
| `tests/test_erp_mock.py` | 22 | PO data, 3-way match, variance, key scanning, error handling |
| `tests/test_vector_db.py` | 22 | Embeddings, cosine similarity, vector DB init, smart match |
| `tests/test_ocr_engine.py` | 12 | System prompt, extraction, HTTP/JSON failures |
| `tests/test_processor.py` | 30 | Audit, fuzzy match, validation, full pipeline |
| `tests/test_integration.py` | 22 | E2E pipeline, error propagation, vendor fallback, 3-way match, duplicates |
| **Total** | **108** | |

Coverage (business logic): `erp_mock` 100%, `ocr_engine` 100%, `vector_db` 100%, `processor` 96%.

`app.py` (Streamlit UI) has 0% coverage — it requires a running Streamlit server and is not unit-testable without a browser harness.

---

## Project Structure

```
invoice-pipeline/
├── app.py                 # Streamlit dashboard (upload, batch process, review)
├── ocr_engine.py          # Vision-LLM OCR via Ollama (llama3.2-vision)
├── processor.py           # Pipeline orchestrator (vendor match, validate, audit, 3-way)
├── vector_db.py           # Semantic vendor matching (embeddings + cosine similarity)
├── erp_mock.py            # Mock ERP/PO database and 3-way match logic
├── conftest.py            # Shared pytest fixtures and PROCESSED_DB auto-reset
├── setup.cfg              # pytest configuration
├── requirements.txt       # Runtime dependencies
├── tests/
│   ├── __init__.py
│   ├── test_erp_mock.py
│   ├── test_ocr_engine.py
│   ├── test_processor.py
│   ├── test_vector_db.py
│   └── test_integration.py
└── README.md
```

---

## Limitations

1. **Ollama required at runtime** — All LLM calls target `http://localhost:11434`. There is no fallback if Ollama is not running; OCR returns a generic error dict, audit returns score 0, vendor matching returns "Unknown".
2. **In-memory state only** — `PROCESSED_DB` (duplicate tracking) and `VENDOR_EMBEDDINGS` (vector cache) are Python globals. They reset on every process restart. There is no persistent database.
3. **Mock PO data** — `erp_mock.POS` contains only two hardcoded purchase orders (PO-1001, PO-1002). There is no real ERP integration.
4. **No authentication** — The Streamlit app has no login or access control.
5. **Single-process** — Duplicate detection relies on a module-level DataFrame; it does not work across multiple workers or server restarts.
6. **`app.py` untestable** — The Streamlit UI layer (0% coverage) cannot be unit-tested without a browser automation harness.
7. **Unused dependencies** — `watchdog` and `plotly` are listed in `requirements.txt` but not imported anywhere in the codebase.
8. **No PDF text extraction** — `ocr_engine.py` sends the raw file bytes as a base64 image. It does not extract text from PDFs; it relies entirely on the vision model reading the rendered image.
9. **Hardcoded thresholds** — Vendor match similarity (0.6), fuzzy match confidence (80%), high-value flag ($10,000), and PO variance tolerance ($1.00) are all hardcoded constants, not configurable via environment variables.
10. **No CI/CD** — There is no GitHub Actions workflow, Makefile, or similar automation. Tests must be run manually.
