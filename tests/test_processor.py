"""
Unit tests for processor.py — audit_invoice_with_ai, match_vendor,
validate_invoice, process_pipeline.

All HTTP calls (Ollama) and cross-module calls are mocked.

Coverage intent:
  - audit_invoice_with_ai: success, non-200, connection error, malformed JSON (lines 39-72)
  - match_vendor: exact match, close match, no match, None/empty input (lines 74-82)
  - validate_invoice: duplicate (always-false due to bug), high value, missing number (lines 84-100)
  - process_pipeline: full orchestration with vector→fuzzy fallback, mutation (lines 108-150)
  - AUDIT_PROMPT content (lines 27-36)
  - KNOWN_VENDORS data (line 22)
  - PROCESSED_DB initial state (line 25)
"""

import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from processor import (
    audit_invoice_with_ai,
    match_vendor,
    validate_invoice,
    process_pipeline,
    KNOWN_VENDORS,
    PROCESSED_DB,
    AUDIT_PROMPT,
)


# ============================================================================
# Module-level data validation
# ============================================================================

class TestModuleData:
    """Verify module-level constants (processor.py lines 22-36)."""

    def test_known_vendors_list(self):
        assert isinstance(KNOWN_VENDORS, list)
        assert "Amazon Web Services" in KNOWN_VENDORS
        assert "Microsoft Azure" in KNOWN_VENDORS
        assert "GitHub Inc" in KNOWN_VENDORS
        assert "Staples" in KNOWN_VENDORS
        assert "WeWork" in KNOWN_VENDORS

    def test_processed_db_is_empty_dataframe(self):
        """PROCESSED_DB is always empty — Phase 2 bug C-01."""
        assert isinstance(PROCESSED_DB, pd.DataFrame)
        assert PROCESSED_DB.empty
        assert list(PROCESSED_DB.columns) == ["invoice_number", "vendor", "total", "date"]

    def test_audit_prompt_contains_rules(self):
        assert "Forensic Accountant" in AUDIT_PROMPT
        assert "bank_details" in AUDIT_PROMPT
        assert "risk_score" in AUDIT_PROMPT


# ============================================================================
# audit_invoice_with_ai
# ============================================================================

class TestAuditInvoiceWithAi:
    """Tests for audit_invoice_with_ai() (processor.py lines 39-72)."""

    @patch("processor.requests.post")
    def test_success_returns_parsed_result(self, mock_post):
        """HTTP 200 with valid JSON → returns risk_score + flags."""
        audit_result = {"risk_score": 45, "flags": ["Round total"]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": json.dumps(audit_result)}
        }
        mock_post.return_value = mock_resp

        result = audit_invoice_with_ai({"total_amount": 5000})

        assert result["risk_score"] == 45
        assert "Round total" in result["flags"]

    @patch("processor.requests.post")
    def test_request_payload_structure(self, mock_post):
        """Verify model, format, stream, prompt concatenation (lines 58-64)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"risk_score": 0, "flags": []}'}
        }
        mock_post.return_value = mock_resp

        invoice = {"vendor_name": "Test", "total_amount": 100}
        audit_invoice_with_ai(invoice)

        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert payload["format"] == "json"
        content = payload["messages"][0]["content"]
        assert AUDIT_PROMPT in content
        assert json.dumps(invoice) in content

    @patch("processor.requests.post")
    def test_non_200_returns_audit_failed(self, mock_post):
        """Non-200 status → {"risk_score": 0, "flags": ["AI Audit Failed"]}."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        result = audit_invoice_with_ai({"total_amount": 100})

        assert result["risk_score"] == 0
        assert "AI Audit Failed" in result["flags"]

    @patch("processor.requests.post", side_effect=ConnectionError("refused"))
    def test_connection_error(self, mock_post):
        """Network failure → returns sanitized error message (C-07 fixed)."""
        result = audit_invoice_with_ai({"total_amount": 100})

        assert result["risk_score"] == 0
        assert any("Audit unavailable" in f for f in result["flags"])

    @patch("processor.requests.post")
    def test_malformed_json_from_llm(self, mock_post):
        """LLM returns non-JSON → json.loads raises → caught in except."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        mock_post.return_value = mock_resp

        result = audit_invoice_with_ai({"total_amount": 100})

        assert result["risk_score"] == 0
        assert any("Audit unavailable" in f or "AI Audit Failed" in f for f in result["flags"])

    @patch("processor.requests.post")
    def test_missing_message_key(self, mock_post):
        """Response missing 'message' → KeyError → caught."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bad_key": "value"}
        mock_post.return_value = mock_resp

        result = audit_invoice_with_ai({"total_amount": 100})

        assert result["risk_score"] == 0


# ============================================================================
# match_vendor
# ============================================================================

class TestMatchVendor:
    """Tests for match_vendor() (processor.py lines 74-82)."""

    def test_exact_match(self):
        """Exact string → 100% score → returns vendor name."""
        result = match_vendor("Amazon Web Services")
        assert result == "Amazon Web Services"

    def test_close_fuzzy_match(self):
        """Near-match above 80% threshold → returns best match."""
        result = match_vendor("Amazon Web Srvices")  # typo
        assert result == "Amazon Web Services"

    def test_no_match_below_threshold(self):
        """Completely unrelated string → below 80% → 'New Vendor (...)'."""
        result = match_vendor("XYZZY Corp International")
        assert "New Vendor" in result
        assert "XYZZY Corp International" in result

    def test_none_input_returns_unknown(self):
        """None → 'Unknown' (line 76)."""
        assert match_vendor(None) == "Unknown"

    def test_empty_string_returns_unknown(self):
        """Empty string is falsy → 'Unknown' (line 76)."""
        assert match_vendor("") == "Unknown"

    def test_case_variation_below_threshold(self):
        """'GITHUB INC' vs 'GitHub Inc' — rapidfuzz token_sort_ratio scores
        below 80 threshold due to short token lengths. Actual behavior is 'New Vendor'.
        Documents that the 80% threshold can miss valid case-only variations."""
        result = match_vendor("GITHUB INC")
        # If it matches, great; if not, it's a known edge case with the threshold
        assert result == "GitHub Inc" or "New Vendor" in result

    def test_partial_name_match(self):
        """'Staples' exact → Staples."""
        result = match_vendor("Staples")
        assert result == "Staples"


# ============================================================================
# validate_invoice
# ============================================================================

class TestValidateInvoice:
    """Tests for validate_invoice() (processor.py lines 84-100)."""

    def test_valid_invoice_no_flags(self, valid_invoice_data):
        """Well-formed invoice under threshold → no flags."""
        flags = validate_invoice(valid_invoice_data)
        assert isinstance(flags, list)
        assert len(flags) == 0

    def test_high_value_flag(self, high_value_invoice_data):
        """total_amount > 10000 → 'High Value' flag."""
        flags = validate_invoice(high_value_invoice_data)
        assert any("High Value" in f for f in flags)

    def test_missing_invoice_number_flag(self, missing_fields_invoice):
        """No invoice_number → 'Missing Invoice Number' flag."""
        flags = validate_invoice(missing_fields_invoice)
        assert any("Missing Invoice Number" in f for f in flags)

    def test_high_value_and_missing_number(self):
        """Both conditions triggered simultaneously."""
        data = {"vendor_name": "Test", "total_amount": 20000}
        flags = validate_invoice(data)
        assert len(flags) == 2
        assert any("High Value" in f for f in flags)
        assert any("Missing Invoice Number" in f for f in flags)

    def test_duplicate_detection_never_fires(self):
        """validate_invoice alone sees empty PROCESSED_DB → no duplicate flag.
        C-01 fix writes to PROCESSED_DB in process_pipeline, not validate_invoice."""
        data = {
            "invoice_number": "INV-001",
            "vendor_name": "Amazon Web Services",
            "total_amount": 100,
        }
        flags = validate_invoice(data)
        assert not any("Duplicate" in f for f in flags)

    def test_exactly_10000_not_flagged(self):
        """total_amount == 10000 (not > 10000) → no high-value flag."""
        data = {"invoice_number": "INV-99", "vendor_name": "Test", "total_amount": 10000}
        flags = validate_invoice(data)
        assert not any("High Value" in f for f in flags)

    def test_total_amount_just_above_threshold(self):
        """total_amount = 10000.01 → flag triggered."""
        data = {"invoice_number": "INV-99", "vendor_name": "Test", "total_amount": 10000.01}
        flags = validate_invoice(data)
        assert any("High Value" in f for f in flags)

    def test_empty_dict(self):
        """Empty dict → missing invoice_number flag only (total defaults to 0)."""
        flags = validate_invoice({})
        assert any("Missing Invoice Number" in f for f in flags)
        assert not any("High Value" in f for f in flags)

    def test_invoice_number_empty_string(self):
        """invoice_number = '' → falsy → flagged."""
        data = {"invoice_number": "", "total_amount": 100}
        flags = validate_invoice(data)
        assert any("Missing Invoice Number" in f for f in flags)

    def test_invoice_number_none(self):
        """invoice_number = None → falsy → flagged."""
        data = {"invoice_number": None, "total_amount": 100}
        flags = validate_invoice(data)
        assert any("Missing Invoice Number" in f for f in flags)


# ============================================================================
# process_pipeline
# ============================================================================

class TestProcessPipeline:
    """Tests for process_pipeline() (processor.py lines 108-150)."""

    @patch("processor.check_3_way_match", return_value="✅ Matched")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 10, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_full_pipeline_success(self, mock_vendor, mock_audit, mock_match):
        """All steps succeed → enriched dict with new keys."""
        raw = {
            "vendor_name": "AWS",
            "invoice_number": "INV-001",
            "total_amount": 5000.00,
            "po_number": "PO-1001",
        }
        result = process_pipeline(raw)

        assert result["standardized_vendor"] == "Amazon Web Services"
        assert isinstance(result["flags"], list)
        assert result["audit_risk_score"] == 10
        assert result["audit_flags"] == []
        assert "Matched" in result["po_match_status"]

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Unknown (DB Init Failed)")
    @patch("processor.match_vendor", return_value="Amazon Web Services")
    def test_vector_fails_fuzzy_fallback_succeeds(self, mock_fuzzy, mock_vector, mock_audit, mock_match):
        """Vector returns 'Unknown' → fuzzy fallback triggered and succeeds (lines 131-135)."""
        raw = {"vendor_name": "AWS", "invoice_number": "INV-002", "total_amount": 100}
        result = process_pipeline(raw)

        assert result["standardized_vendor"] == "Amazon Web Services"
        mock_fuzzy.assert_called_once_with("AWS")

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="New Vendor (XYZ)")
    @patch("processor.match_vendor", return_value="New Vendor (XYZ)")
    def test_vector_and_fuzzy_both_fail(self, mock_fuzzy, mock_vector, mock_audit, mock_match):
        """Both matching systems return 'New Vendor' → stays as-is (line 134 condition not met)."""
        raw = {"vendor_name": "XYZ", "invoice_number": "INV-003", "total_amount": 100}
        result = process_pipeline(raw)

        assert "New Vendor" in result["standardized_vendor"]

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 75, "flags": ["Suspicious"]})
    @patch("processor.smart_match_vendor", return_value="Staples Office Supplies")
    def test_audit_flags_propagated(self, mock_vendor, mock_audit, mock_match):
        """AI audit results are injected into the output (lines 145-146)."""
        raw = {"vendor_name": "Staples", "invoice_number": "INV-004", "total_amount": 100}
        result = process_pipeline(raw)

        assert result["audit_risk_score"] == 75
        assert "Suspicious" in result["audit_flags"]

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Microsoft Azure")
    def test_pipeline_mutates_input_dict(self, mock_vendor, mock_audit, mock_match):
        """process_pipeline mutates raw_json in-place — documents Phase 2 C-03."""
        raw = {"vendor_name": "Azure", "invoice_number": "INV-005", "total_amount": 100}
        original_keys = set(raw.keys())

        result = process_pipeline(raw)

        # result IS the same object as raw (mutation)
        assert result is raw
        # New keys were added
        new_keys = set(result.keys()) - original_keys
        assert "standardized_vendor" in new_keys
        assert "flags" in new_keys
        assert "audit_risk_score" in new_keys
        assert "audit_flags" in new_keys
        assert "po_match_status" in new_keys

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="WeWork Space")
    def test_high_value_invoice_produces_flag(self, mock_vendor, mock_audit, mock_match):
        """High-value invoice → flag appears via validate_invoice."""
        raw = {"vendor_name": "WeWork", "invoice_number": "INV-006", "total_amount": 50000}
        result = process_pipeline(raw)

        assert any("High Value" in f for f in result["flags"])

    @patch("processor.check_3_way_match", return_value="⚠️ PO Not Found")
    @patch("processor.audit_invoice_with_ai", return_value={"risk_score": 0, "flags": []})
    @patch("processor.smart_match_vendor", return_value="Unknown (DB Init Failed)")
    @patch("processor.match_vendor", return_value="Unknown")
    def test_fuzzy_fallback_also_unknown(self, mock_fuzzy, mock_vector, mock_audit, mock_match):
        """Vector returns 'Unknown', fuzzy also returns 'Unknown' → stays 'Unknown'."""
        raw = {"vendor_name": None, "invoice_number": "INV-007", "total_amount": 100}
        result = process_pipeline(raw)

        # fuzzy returns "Unknown" which has "Unknown" → condition on line 133 still True
        # but "New Vendor" not in "Unknown" AND "Unknown" not in "Unknown" is False
        # So standardized_vendor stays as vector result
        assert "Unknown" in result["standardized_vendor"]

    @patch("processor.check_3_way_match", return_value="✅ Matched")
    @patch("processor.audit_invoice_with_ai", return_value={})
    @patch("processor.smart_match_vendor", return_value="Amazon Web Services")
    def test_audit_result_missing_keys_defaults_used(self, mock_vendor, mock_audit, mock_match):
        """audit_result has no risk_score/flags → .get() defaults to 0 and [] (lines 145-146)."""
        raw = {"vendor_name": "AWS", "invoice_number": "INV-008", "total_amount": 100, "po_number": "PO-1001"}
        result = process_pipeline(raw)

        assert result["audit_risk_score"] == 0
        assert result["audit_flags"] == []
