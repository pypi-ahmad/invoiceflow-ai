# invoiceflow-ai

## Overview

An automated invoice processing pipeline built on local Large Language Models (via [Ollama](https://ollama.com)). The system extracts structured data from invoice images using a Vision LLM, normalizes vendor names through semantic and fuzzy matching, applies rule-based validation, performs AI-driven fraud detection, and cross-references invoices against purchase orders. A Streamlit dashboard provides batch upload, analytics, and human review.

## Tech Stack

- Python (requirements.txt based)

## Repository Structure

- `.gitignore`
- `app.py`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `conftest.py`
- `CONTRIBUTING.md`
- `erp_mock.py`
- `LICENSE`
- `ocr_engine.py`
- `processor.py`
- `README.md`
- `requirements.txt`
- ... and 5 more entries

## Getting Started

### Prerequisites

- Git
- Runtime dependencies for this project's stack

### Installation

```bash
uv venv
uv pip install -r requirements.txt
```

## Usage

Run the primary app with `uv run app.py`.

## Testing

Run tests with `uv run pytest` from repository root.

## Security

Please review [SECURITY.md](SECURITY.md) for reporting and handling security issues.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening issues or pull requests.

## Changelog

Ongoing changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
