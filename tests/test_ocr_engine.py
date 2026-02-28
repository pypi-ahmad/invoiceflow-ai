"""
Unit tests for ocr_engine.py — extract_invoice_data().

All HTTP calls to Ollama and file I/O are mocked — no real network or disk needed.

Coverage intent:
  - File read + base64 encoding (lines 48-49)
  - HTTP POST to Ollama /api/chat with correct payload (lines 56-65)
  - Successful response parsing: strip markdown fences, json.loads (line 71)
  - raise_for_status triggers HTTPError → caught → error dict (line 67)
  - Malformed JSON response from LLM → JSONDecodeError → error dict
  - File not found → error dict (line 74-75)
  - Connection refused → error dict
  - SYSTEM_PROMPT content integrity (lines 17-27)
"""

import pytest
import json
import base64
from unittest.mock import patch, mock_open, MagicMock
from ocr_engine import extract_invoice_data, SYSTEM_PROMPT


# ============================================================================
# SYSTEM_PROMPT validation
# ============================================================================

class TestSystemPrompt:
    """Verify SYSTEM_PROMPT structure (ocr_engine.py lines 17-27)."""

    def test_prompt_mentions_required_fields(self):
        assert "vendor_name" in SYSTEM_PROMPT
        assert "invoice_number" in SYSTEM_PROMPT
        assert "invoice_date" in SYSTEM_PROMPT
        assert "total_amount" in SYSTEM_PROMPT
        assert "currency" in SYSTEM_PROMPT
        assert "line_items" in SYSTEM_PROMPT

    def test_prompt_requests_json_only(self):
        assert "JSON" in SYSTEM_PROMPT
        assert "No markdown" in SYSTEM_PROMPT


# ============================================================================
# extract_invoice_data — Happy Path
# ============================================================================

class TestExtractInvoiceDataHappyPath:
    """Tests for successful extraction (ocr_engine.py lines 44-72)."""

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_image_bytes"))
    def test_successful_extraction(self, mock_post):
        """Valid response → parsed dict with expected fields."""
        llm_response = {
            "vendor_name": "Staples",
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "total_amount": 259.99,
            "currency": "USD",
            "line_items": []
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": json.dumps(llm_response)}
        }
        mock_post.return_value = mock_resp

        result = extract_invoice_data("fake_image.png")

        assert result["vendor_name"] == "Staples"
        assert result["invoice_number"] == "INV-001"
        assert result["total_amount"] == 259.99

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"fake_image_bytes"))
    def test_response_with_markdown_fences_stripped(self, mock_post):
        """LLM wraps JSON in ```json ... ``` → fences are stripped (line 71)."""
        llm_json = {"vendor_name": "Test", "total_amount": 100}
        wrapped = f"```json\n{json.dumps(llm_json)}\n```"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": wrapped}}
        mock_post.return_value = mock_resp

        result = extract_invoice_data("test.png")
        assert result["vendor_name"] == "Test"

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"\x89PNG\r\n"))
    def test_base64_encoding_in_payload(self, mock_post):
        """Image bytes are base64-encoded in the API payload (lines 48-49, 62)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": '{"vendor_name": "X"}'}
        }
        mock_post.return_value = mock_resp

        extract_invoice_data("image.png")

        call_json = mock_post.call_args[1]["json"]
        expected_b64 = base64.b64encode(b"\x89PNG\r\n").decode("utf-8")
        assert call_json["messages"][0]["images"][0] == expected_b64

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_request_payload_structure(self, mock_post):
        """Verify full payload structure: model, messages, stream (lines 57-65)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"a": 1}'}}
        mock_post.return_value = mock_resp

        extract_invoice_data("img.png")

        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3.2-vision"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == SYSTEM_PROMPT


# ============================================================================
# extract_invoice_data — Failure Paths
# ============================================================================

class TestExtractInvoiceDataFailure:
    """Tests for error handling (ocr_engine.py lines 67, 74-75)."""

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_http_error_returns_error_dict(self, mock_post):
        """raise_for_status raises HTTPError → caught → error dict."""
        import requests as real_requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = real_requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp

        result = extract_invoice_data("img.png")

        assert "error" in result
        assert result["error"] == "OCR processing failed"
        assert result["file"] == "img.png"

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_malformed_json_response(self, mock_post):
        """LLM returns invalid JSON → json.loads fails → error dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": "NOT VALID JSON {{"}}
        mock_post.return_value = mock_resp

        result = extract_invoice_data("test.png")

        assert "error" in result
        assert result["file"] == "test.png"

    def test_file_not_found(self):
        """Nonexistent file → FileNotFoundError → caught → error dict."""
        result = extract_invoice_data("nonexistent_file_12345.png")

        assert "error" in result
        assert result["file"] == "nonexistent_file_12345.png"

    @patch("ocr_engine.requests.post", side_effect=ConnectionError("Connection refused"))
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_connection_refused(self, mock_post):
        """Ollama not running → ConnectionError → sanitized error dict (C-07)."""
        result = extract_invoice_data("img.png")

        assert "error" in result
        assert result["error"] == "OCR processing failed"

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_missing_message_key_in_response(self, mock_post):
        """Response JSON missing 'message' key → KeyError → error dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"unexpected_key": "value"}
        mock_post.return_value = mock_resp

        result = extract_invoice_data("img.png")

        assert "error" in result

    @patch("ocr_engine.requests.post")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_missing_content_key_in_message(self, mock_post):
        """Response has 'message' but no 'content' → KeyError → error dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {}}
        mock_post.return_value = mock_resp

        result = extract_invoice_data("img.png")

        assert "error" in result
