"""
Unit tests for vector_db.py — get_embedding, cosine_similarity,
initialize_vector_db, smart_match_vendor.

All HTTP calls to Ollama are mocked — no network required.

Coverage intent:
  - get_embedding: success, non-200 fallback, connection error (lines 28-56)
  - cosine_similarity: normal, orthogonal, identical, zero-vector (line 57)
  - initialize_vector_db: populates VENDOR_EMBEDDINGS (lines 59-72)
  - smart_match_vendor: None/empty input, DB init failure, embedding failure,
    above-threshold match, below-threshold → New Vendor (lines 74-117)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import vector_db
from vector_db import (
    get_embedding,
    cosine_similarity,
    initialize_vector_db,
    smart_match_vendor,
    VENDORS,
    VENDOR_EMBEDDINGS,
)


# ============================================================================
# Module-level data
# ============================================================================

class TestModuleData:
    """Verify VENDORS list structure (vector_db.py lines 19-22)."""

    def test_vendors_is_nonempty_list(self):
        assert isinstance(VENDORS, list)
        assert len(VENDORS) == 7

    def test_known_vendors_present(self):
        assert "Amazon Web Services" in VENDORS
        assert "Microsoft Azure" in VENDORS
        assert "Google Cloud Platform" in VENDORS


# ============================================================================
# get_embedding
# ============================================================================

class TestGetEmbedding:
    """Tests for get_embedding() (vector_db.py lines 28-56)."""

    @patch("vector_db.requests.post")
    def test_success_returns_embedding(self, mock_post, sample_embedding):
        """HTTP 200 with valid embedding → returns list of floats."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": sample_embedding}
        mock_post.return_value = mock_resp

        result = get_embedding("test text", model="embeddinggemma")

        assert result == sample_embedding
        mock_post.assert_called_once()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "embeddinggemma"
        assert call_json["prompt"] == "test text"

    @patch("vector_db.requests.post")
    def test_success_missing_embedding_key(self, mock_post):
        """HTTP 200 but response JSON has no 'embedding' key → returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        result = get_embedding("test", model="all-minilm")
        assert result is None

    @patch("vector_db.requests.post")
    def test_non_200_triggers_fallback_to_allminilm(self, mock_post, sample_embedding):
        """Non-200 with model != 'all-minilm' → recursive call with 'all-minilm'."""
        # First call (embeddinggemma) returns 404, second call (all-minilm) returns 200
        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {"embedding": sample_embedding}

        mock_post.side_effect = [mock_resp_404, mock_resp_200]

        result = get_embedding("test", model="embeddinggemma")

        assert result == sample_embedding
        assert mock_post.call_count == 2
        # Second call should use all-minilm
        second_call_json = mock_post.call_args_list[1][1]["json"]
        assert second_call_json["model"] == "all-minilm"

    @patch("vector_db.requests.post")
    def test_non_200_allminilm_returns_none(self, mock_post):
        """Non-200 with model = 'all-minilm' → no further fallback → None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        result = get_embedding("test", model="all-minilm")
        assert result is None
        assert mock_post.call_count == 1  # No recursion

    @patch("vector_db.requests.post", side_effect=ConnectionError("refused"))
    def test_connection_error_returns_none(self, mock_post):
        """Network failure → caught by except → returns None."""
        result = get_embedding("test", model="embeddinggemma")
        assert result is None

    @patch("vector_db.requests.post")
    def test_api_url_is_correct(self, mock_post):
        """Verify the endpoint URL (vector_db.py line 38)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [0.1]}
        mock_post.return_value = mock_resp

        get_embedding("hello")
        url_called = mock_post.call_args[0][0]
        assert url_called == "http://localhost:11434/api/embeddings"


# ============================================================================
# cosine_similarity
# ============================================================================

class TestCosineSimilarity:
    """Tests for cosine_similarity() (vector_db.py line 57)."""

    def test_identical_vectors(self):
        """Identical vectors → similarity = 1.0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors → similarity = 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite direction → similarity = -1.0."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_different_magnitude_same_direction(self):
        """Parallel vectors with different magnitude → still 1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [2.0, 4.0, 6.0]
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_zero_vector_raises_or_nan(self):
        """Zero vector causes division by zero → guarded to return 0.0 (Phase 2 M-04 fixed)."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        result = cosine_similarity(v1, v2)
        assert result == 0.0

    def test_numpy_arrays_accepted(self):
        """Function should work with numpy arrays too."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)


# ============================================================================
# initialize_vector_db
# ============================================================================

class TestInitializeVectorDb:
    """Tests for initialize_vector_db() (vector_db.py lines 59-72)."""

    def setup_method(self):
        """Clear global state before each test."""
        VENDOR_EMBEDDINGS.clear()

    def teardown_method(self):
        """Clear global state after each test."""
        VENDOR_EMBEDDINGS.clear()

    @patch("vector_db.get_embedding")
    def test_all_vendors_embedded(self, mock_get_emb):
        """When all embeddings succeed, all vendors are in VENDOR_EMBEDDINGS."""
        mock_get_emb.return_value = [0.1, 0.2, 0.3]

        initialize_vector_db()

        assert len(VENDOR_EMBEDDINGS) == len(VENDORS)
        for v in VENDORS:
            assert v in VENDOR_EMBEDDINGS

    @patch("vector_db.get_embedding")
    def test_calls_with_embeddinggemma_model(self, mock_get_emb):
        """Verify model='embeddinggemma' is passed (line 65)."""
        mock_get_emb.return_value = [0.1]

        initialize_vector_db()

        for call in mock_get_emb.call_args_list:
            assert call[1].get("model", call[0][1] if len(call[0]) > 1 else None) == "embeddinggemma"

    @patch("vector_db.get_embedding", return_value=None)
    def test_failed_embeddings_not_stored(self, mock_get_emb):
        """When get_embedding returns None, vendor is skipped."""
        initialize_vector_db()

        assert len(VENDOR_EMBEDDINGS) == 0

    @patch("vector_db.get_embedding")
    def test_partial_failure(self, mock_get_emb):
        """Some embeddings succeed, others fail → only successful ones stored."""
        call_count = 0

        def side_effect(text, model=None):
            nonlocal call_count
            call_count += 1
            return [0.1, 0.2] if call_count % 2 == 0 else None

        mock_get_emb.side_effect = side_effect

        initialize_vector_db()

        # Only even-numbered calls succeed
        assert len(VENDOR_EMBEDDINGS) < len(VENDORS)
        assert len(VENDOR_EMBEDDINGS) > 0


# ============================================================================
# smart_match_vendor
# ============================================================================

class TestSmartMatchVendor:
    """Tests for smart_match_vendor() (vector_db.py lines 74-117)."""

    def setup_method(self):
        """Clear global state before each test."""
        VENDOR_EMBEDDINGS.clear()

    def teardown_method(self):
        """Clear global state after each test."""
        VENDOR_EMBEDDINGS.clear()

    def test_none_input_returns_unknown(self):
        """None → 'Unknown' (line 87)."""
        assert smart_match_vendor(None) == "Unknown"

    def test_empty_string_returns_unknown(self):
        """Empty string is falsy → 'Unknown' (line 87)."""
        assert smart_match_vendor("") == "Unknown"

    @patch("vector_db.initialize_vector_db")
    def test_empty_db_after_init_returns_db_init_failed(self, mock_init):
        """VENDOR_EMBEDDINGS empty after init call → 'Unknown (DB Init Failed)' (line 93)."""
        # mock_init does nothing so VENDOR_EMBEDDINGS stays empty
        result = smart_match_vendor("Amazon")
        assert "DB Init Failed" in result
        mock_init.assert_called_once()

    @patch("vector_db.get_embedding", return_value=None)
    def test_embedding_failure_returns_embedding_failed(self, mock_get_emb):
        """Input embedding fails → 'Unknown (Embedding Failed)' (line 101)."""
        # Pre-populate VENDOR_EMBEDDINGS so we skip init
        VENDOR_EMBEDDINGS["Amazon Web Services"] = [0.1, 0.2, 0.3]

        result = smart_match_vendor("Amazon")
        assert "Embedding Failed" in result

    @patch("vector_db.get_embedding")
    def test_high_similarity_returns_best_match(self, mock_get_emb):
        """Score > 0.6 → return vendor name (line 113)."""
        # Pre-populate: two vendors with known embeddings
        VENDOR_EMBEDDINGS["Amazon Web Services"] = [1.0, 0.0, 0.0]
        VENDOR_EMBEDDINGS["Microsoft Azure"] = [0.0, 1.0, 0.0]

        # Input embedding is very close to Amazon
        mock_get_emb.return_value = [0.95, 0.05, 0.0]

        result = smart_match_vendor("AWS")
        assert result == "Amazon Web Services"

    @patch("vector_db.get_embedding")
    def test_low_similarity_returns_new_vendor(self, mock_get_emb):
        """Score ≤ 0.6 for all vendors → 'New Vendor (...)' (line 116)."""
        # Orthogonal embeddings ensure low similarity
        VENDOR_EMBEDDINGS["Amazon Web Services"] = [1.0, 0.0, 0.0]
        VENDOR_EMBEDDINGS["Microsoft Azure"] = [0.0, 1.0, 0.0]

        # Input embedding is orthogonal to both
        mock_get_emb.return_value = [0.0, 0.0, 1.0]

        result = smart_match_vendor("Totally Unknown Co")
        assert "New Vendor" in result
        assert "Totally Unknown Co" in result

    @patch("vector_db.get_embedding")
    def test_lazy_initialization_triggered(self, mock_get_emb):
        """When VENDOR_EMBEDDINGS is empty, initialize_vector_db is called (line 91)."""
        # get_embedding will populate during init AND serve the input query
        call_count = 0

        def side_effect(text, model=None):
            nonlocal call_count
            call_count += 1
            if text == "test_input":
                return [1.0, 0.0, 0.0]
            return [1.0, 0.0, 0.0]  # Same direction for simplicity

        mock_get_emb.side_effect = side_effect

        result = smart_match_vendor("test_input")
        # Should have called get_embedding for each vendor (init) + 1 for input
        assert call_count == len(VENDORS) + 1

    @patch("vector_db.get_embedding")
    def test_best_match_among_multiple(self, mock_get_emb):
        """Returns the vendor with highest cosine similarity."""
        VENDOR_EMBEDDINGS["Vendor A"] = [1.0, 0.0, 0.0]
        VENDOR_EMBEDDINGS["Vendor B"] = [0.7, 0.7, 0.0]
        VENDOR_EMBEDDINGS["Vendor C"] = [0.0, 0.0, 1.0]

        # Input closest to Vendor B
        mock_get_emb.return_value = [0.6, 0.8, 0.0]

        result = smart_match_vendor("some vendor")
        assert result == "Vendor B"
