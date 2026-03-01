"""Tests for InductiveLayer — HDC inductive reasoning."""

import numpy as np
import pytest

from glyphh.state import InductiveLayer, Centroid


# ── Construction ──

class TestConstruction:
    def test_default_params(self):
        layer = InductiveLayer()
        assert layer._dim == 10000
        assert layer._seed == 97
        assert layer._min_confidence == 0.05
        assert layer.centroids == {}
        assert layer.labels == []

    def test_custom_params(self):
        layer = InductiveLayer(dimension=5000, seed=42, min_confidence=0.10)
        assert layer._dim == 5000
        assert layer._seed == 42
        assert layer._min_confidence == 0.10


# ── Feature Encoding ──

class TestEncoding:
    def test_encode_produces_bipolar(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        vec = layer._encode_features({"query": "hello world"})
        assert vec.dtype == np.int8
        assert set(np.unique(vec)).issubset({-1, 1})
        assert len(vec) == 1000

    def test_same_features_same_vector(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        v1 = layer._encode_features({"query": "hello world", "state": "home"})
        v2 = layer._encode_features({"query": "hello world", "state": "home"})
        assert np.array_equal(v1, v2)

    def test_different_features_different_vectors(self):
        from glyphh.core.ops import cosine_similarity
        layer = InductiveLayer(dimension=10000, seed=97)
        v1 = layer._encode_features({"query": "grep budget", "state": "/workspace"})
        v2 = layer._encode_features({"query": "sort file", "state": "/tmp"})
        sim = cosine_similarity(v1, v2)
        # Different features → low cosine (quasi-orthogonal)
        assert abs(sim) < 0.35

    def test_shared_words_positive_similarity(self):
        from glyphh.core.ops import cosine_similarity
        layer = InductiveLayer(dimension=10000, seed=97)
        v1 = layer._encode_features({"query": "grep budget analysis report"})
        v2 = layer._encode_features({"query": "grep budget financial data"})
        sim = cosine_similarity(v1, v2)
        # Shared "grep" and "budget" → positive similarity
        assert sim > 0.0

    def test_empty_value_skipped(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        v1 = layer._encode_features({"query": "hello", "empty": ""})
        v2 = layer._encode_features({"query": "hello"})
        # Empty value is skipped, so result should be the same
        assert np.array_equal(v1, v2)

    def test_empty_features_returns_vector(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        vec = layer._encode_features({})
        assert vec.dtype == np.int8
        assert len(vec) == 1000


# ── Learn ──

class TestLearn:
    def test_learn_creates_centroid(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "label_a")
        assert "label_a" in layer.centroids
        assert layer.centroids["label_a"].count == 1
        assert layer.centroids["label_a"].strength == 1.0

    def test_learn_increments_count(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "label_a")
        layer.learn({"query": "world"}, "label_a")
        layer.learn({"query": "foo"}, "label_a")
        assert layer.centroids["label_a"].count == 3

    def test_learn_multiple_labels(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        layer.learn({"query": "world"}, "no")
        assert set(layer.labels) == {"yes", "no"}

    def test_centroid_is_bipolar(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "label_a")
        layer.learn({"query": "world"}, "label_a")
        vec = layer.centroids["label_a"].vector
        assert set(np.unique(vec)).issubset({-1, 1})


# ── Predict ──

class TestPredict:
    def test_predict_empty_returns_none(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        result = layer.predict({"query": "hello"})
        assert result["label"] is None
        assert result["confidence"] == 0.0

    def test_predict_single_centroid(self):
        layer = InductiveLayer(dimension=10000, seed=97)
        layer.learn({"query": "grep budget analysis"}, "cd_needed")
        result = layer.predict({"query": "grep budget analysis"})
        assert result["label"] == "cd_needed"
        assert result["confidence"] > 0

    def test_predict_distinguishes_labels(self):
        layer = InductiveLayer(dimension=10000, seed=97)
        # Learn patterns with different features
        for _ in range(5):
            layer.learn({"query": "grep search find cat", "state": "/other"}, "cd_needed")
            layer.learn({"query": "sort echo wc head", "state": "/here"}, "cd_not_needed")

        # Query similar to cd_needed pattern
        result = layer.predict({"query": "grep search find", "state": "/other"})
        assert result["label"] == "cd_needed"

        # Query similar to cd_not_needed pattern
        result = layer.predict({"query": "sort echo wc", "state": "/here"})
        assert result["label"] == "cd_not_needed"

    def test_predict_returns_scores_for_all_labels(self):
        layer = InductiveLayer(dimension=10000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        layer.learn({"query": "world"}, "no")
        result = layer.predict({"query": "hello"})
        assert "yes" in result["scores"]
        assert "no" in result["scores"]

    def test_predict_below_threshold_returns_none(self):
        layer = InductiveLayer(dimension=10000, seed=97, min_confidence=0.99)
        layer.learn({"query": "hello"}, "yes")
        result = layer.predict({"query": "completely different unrelated"})
        assert result["label"] is None


# ── Confirm (Hebbian) ──

class TestConfirm:
    def test_confirm_strengthens(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        before = layer.centroids["yes"].strength
        layer.confirm(was_correct=True, label="yes")
        after = layer.centroids["yes"].strength
        assert after > before

    def test_confirm_weakens(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        before = layer.centroids["yes"].strength
        layer.confirm(was_correct=False, label="yes")
        after = layer.centroids["yes"].strength
        assert after < before

    def test_strength_bounded_max(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        for _ in range(100):
            layer.confirm(was_correct=True, label="yes")
        assert layer.centroids["yes"].strength <= 3.0

    def test_strength_bounded_min(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        for _ in range(100):
            layer.confirm(was_correct=False, label="yes")
        assert layer.centroids["yes"].strength >= 0.3

    def test_confirm_unknown_noop(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        layer.confirm(was_correct=True, label="nonexistent")
        # No error, no change


# ── Reset ──

class TestReset:
    def test_reset_preserves_centroids(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        layer.learn({"query": "world"}, "no")
        layer.confirm(was_correct=True, label="yes")
        layer.reset()
        assert "yes" in layer.centroids
        assert "no" in layer.centroids

    def test_reset_resets_strengths(self):
        layer = InductiveLayer(dimension=1000, seed=97)
        layer.learn({"query": "hello"}, "yes")
        layer.confirm(was_correct=True, label="yes")
        assert layer.centroids["yes"].strength > 1.0
        layer.reset()
        assert layer.centroids["yes"].strength == 1.0


# ── Integration ──

class TestIntegration:
    def test_filesystem_cd_scenario(self):
        """Simulate the BFCL cd-detection use case."""
        layer = InductiveLayer(dimension=10000, seed=97)

        # Pre-seed: turns where cd was needed
        layer.learn({"query": "grep for budget in the report", "state": "/workspace", "actions": "mkdir mv"}, "cd_needed")
        layer.learn({"query": "find all csv files", "state": "/workspace", "actions": "cp mkdir"}, "cd_needed")
        layer.learn({"query": "search for error logs", "state": "/home", "actions": "mv touch"}, "cd_needed")

        # Pre-seed: turns where cd was NOT needed
        layer.learn({"query": "sort the output file", "state": "/workspace/temp", "actions": "grep"}, "cd_not_needed")
        layer.learn({"query": "count lines in report", "state": "/workspace/docs", "actions": "cat"}, "cd_not_needed")
        layer.learn({"query": "echo hello world", "state": "/home", "actions": "ls"}, "cd_not_needed")

        # Query similar to cd_needed
        result = layer.predict({"query": "grep for analysis in the file", "state": "/workspace", "actions": "mkdir mv"})
        assert result["label"] == "cd_needed"

        # Query similar to cd_not_needed
        result = layer.predict({"query": "sort the results", "state": "/workspace/temp", "actions": "grep"})
        assert result["label"] == "cd_not_needed"

    def test_auth_scenario(self):
        """Domain-agnostic: auth prerequisite detection."""
        layer = InductiveLayer(dimension=10000, seed=97)

        layer.learn({"query": "post a message", "session": "anonymous"}, "login_needed")
        layer.learn({"query": "view dashboard", "session": "authenticated"}, "login_not_needed")

        result = layer.predict({"query": "send a message", "session": "anonymous"})
        assert result["label"] == "login_needed"

    def test_centroid_dataclass(self):
        c = Centroid(label="test", vector=np.ones(100, dtype=np.int8))
        assert c.label == "test"
        assert c.count == 1
        assert c.strength == 1.0
        c.strengthen()
        assert c.strength > 1.0
        c.weaken()
        # Still above 1.0 after one strengthen then one weaken
        assert c.strength >= 0.3


# ── Packs ──

class TestPacks:
    def test_filesystem_pack_loads(self):
        layer = InductiveLayer(dimension=10000, seed=97, packs=["filesystem"])
        assert "cd_needed" in layer.labels
        assert "cd_not_needed" in layer.labels
        # Should have learned from all pack patterns
        assert layer.centroids["cd_needed"].count >= 2
        assert layer.centroids["cd_not_needed"].count >= 2

    def test_pack_predictions_work(self):
        layer = InductiveLayer(dimension=10000, seed=97, packs=["filesystem"])
        # Query similar to cd_needed patterns
        result = layer.predict({"query_tokens": "grep search find files directory"})
        assert result["label"] == "cd_needed"

        # Query similar to cd_not_needed patterns
        result = layer.predict({"query_tokens": "echo print output text"})
        assert result["label"] == "cd_not_needed"

    def test_unknown_pack_raises(self):
        with pytest.raises(FileNotFoundError):
            InductiveLayer(packs=["nonexistent_domain"])
