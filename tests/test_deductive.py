"""Tests for the DeductiveLayer — HDC deductive reasoning (SDK version)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from glyphh.state import DeductiveLayer, Transition
from glyphh.core.ops import cosine_similarity, generate_symbol


# ── Fixtures ──


@pytest.fixture
def layer():
    """DeductiveLayer configured for filesystem (like BFCL model uses it)."""
    dl = DeductiveLayer(dimension=10000, seed=89)
    dl.add_transition(
        name="location_change",
        directing_actions=["mv", "cp", "mkdir", "touch"],
        operating_actions=[
            "grep", "cat", "sort", "wc", "head", "tail", "diff",
            "echo", "find", "ls",
        ],
        prerequisite="cd",
    )
    return dl


@pytest.fixture
def bare_layer():
    """DeductiveLayer with no transitions registered."""
    return DeductiveLayer(dimension=10000, seed=89)


# ── Unit Tests: Construction ──


class TestConstruction:
    def test_default_params(self, bare_layer):
        assert bare_layer._dim == 10000
        assert bare_layer._seed == 89
        assert bare_layer._decay == 0.7
        assert bare_layer.depth == 0

    def test_add_transition(self, bare_layer):
        bare_layer.add_transition(
            name="test_transition",
            directing_actions=["create", "update"],
            operating_actions=["read", "search"],
            prerequisite="prepare",
        )
        assert "test_transition" in bare_layer.transitions
        assert bare_layer.transitions["test_transition"].prerequisite == "prepare"
        assert bare_layer.transitions["test_transition"].strength == 1.0

    def test_multiple_transitions(self, bare_layer):
        bare_layer.add_transition("t1", ["a"], ["b"], "pre1")
        bare_layer.add_transition("t2", ["c"], ["d"], "pre2")
        assert len(bare_layer.transitions) == 2


# ── Unit Tests: Observe ──


class TestObserve:
    def test_observe_increments_depth(self, layer):
        assert layer.depth == 0
        layer.observe(state="/workspace", actions=["ls"])
        assert layer.depth == 1
        layer.observe(state="/workspace/src", actions=["cat"])
        assert layer.depth == 2

    def test_observe_with_targets(self, layer):
        layer.observe(
            state="/workspace",
            actions=["mkdir", "mv"],
            targets=["/workspace/temp"],
        )
        assert layer._last_target == "/workspace/temp"
        assert len(layer._target_history) == 1

    def test_observe_bounds_history(self, layer):
        for i in range(15):
            layer.observe(state=f"/dir_{i}", actions=["ls"])
        assert len(layer._action_history) == 10

    def test_observe_directing_action_without_target(self, layer):
        """Directing actions without explicit targets record current state as target."""
        layer.observe(state="/workspace", actions=["mv"])
        assert len(layer._target_history) == 1


# ── Unit Tests: Deduce ──


class TestDeduce:
    def test_no_deduction_without_observations(self, layer):
        result = layer.deduce("search for files", current_state="/workspace")
        assert result["prerequisites"] == []

    def test_no_deduction_same_state(self, layer):
        """When target matches current state, no prerequisites needed."""
        layer.observe(state="/workspace", actions=["mkdir", "mv"], targets=["/workspace"])
        result = layer.deduce("grep for data", current_state="/workspace")
        assert result["mismatch_score"] <= 0.10

    def test_mismatch_detected_different_state(self, layer):
        """When target differs from current state, mismatch should be detected."""
        # Actions directed files to /workspace/temp
        layer.observe(
            state="/workspace",
            actions=["mkdir", "mv"],
            targets=["/workspace/temp"],
        )
        # But we're still at /workspace — should detect mismatch
        result = layer.deduce("grep for budget analysis", current_state="/workspace")
        assert result["mismatch_score"] > 0.10

    def test_deduction_emits_prerequisite(self, layer):
        """When mismatch + directing history + operating query → emit cd."""
        layer.observe(
            state="/workspace",
            actions=["mkdir", "mv"],
            targets=["/workspace/temp"],
        )
        result = layer.deduce("grep for budget analysis", current_state="/workspace")
        if result["prerequisites"]:
            assert "cd" in result["prerequisites"]
            assert result["confidence"] > 0
            assert result["target"] == "/workspace/temp"

    def test_no_deduction_without_operating_query(self, layer):
        """If query doesn't imply an operating action, no deduction."""
        layer.observe(
            state="/workspace",
            actions=["mkdir", "mv"],
            targets=["/workspace/temp"],
        )
        # "move the file" is a directing action, not operating
        result = layer.deduce("move the document to archive", current_state="/workspace")
        # Should not trigger because the query doesn't have operating verbs
        # (move is not in the registered operating_actions for this transition)
        assert result["prerequisites"] == [] or result["mismatch_score"] <= 0.10

    def test_deduce_returns_correct_structure(self, layer):
        layer.observe(state="/workspace", actions=["ls"])
        result = layer.deduce("list files", current_state="/workspace")
        assert "prerequisites" in result
        assert "mismatch_score" in result
        assert "target" in result
        assert "confidence" in result
        assert isinstance(result["prerequisites"], list)


# ── Unit Tests: Confirm (Hebbian) ──


class TestConfirm:
    def test_confirm_strengthens(self, layer):
        initial = layer.transitions["location_change"].strength
        layer.confirm(True, "cd")
        assert layer.transitions["location_change"].strength > initial

    def test_confirm_weakens_on_incorrect(self, layer):
        initial = layer.transitions["location_change"].strength
        layer.confirm(False, "cd")
        assert layer.transitions["location_change"].strength < initial

    def test_strength_minimum(self, layer):
        for _ in range(50):
            layer.confirm(False, "cd")
        assert layer.transitions["location_change"].strength >= 0.3

    def test_strength_maximum(self, layer):
        for _ in range(100):
            layer.confirm(True, "cd")
        assert layer.transitions["location_change"].strength <= 3.0

    def test_confirm_unknown_noop(self, layer):
        initial = layer.transitions["location_change"].strength
        layer.confirm(True, "nonexistent")
        assert layer.transitions["location_change"].strength == initial

    def test_fire_count_increments(self, layer):
        assert layer.transitions["location_change"].fire_count == 0
        layer.confirm(True, "cd")
        assert layer.transitions["location_change"].fire_count == 1


# ── Unit Tests: Reset ──


class TestReset:
    def test_reset_clears_state(self, layer):
        layer.observe(state="/workspace", actions=["ls"])
        layer.observe(state="/workspace/src", actions=["cat"])
        assert layer.depth == 2
        layer.reset()
        assert layer.depth == 0
        assert layer._state_vector is None
        assert layer._target_history == []
        assert layer._last_target is None

    def test_reset_preserves_transitions(self, layer):
        layer.confirm(True, "cd")
        layer.reset()
        # Transitions persist but strength resets
        assert "location_change" in layer.transitions
        assert layer.transitions["location_change"].strength == 1.0
        assert layer.transitions["location_change"].fire_count == 0


# ── Integration Tests ──


class TestIntegration:
    def test_full_filesystem_cycle(self, layer):
        """Simulate the BFCL cd-missing scenario."""
        # Turn 0: cd to document, mkdir temp, mv file to temp
        layer.observe(
            state="/workspace/document",
            actions=["cd", "mkdir", "mv"],
            targets=["/workspace/document/temp"],
        )

        # Turn 1: query wants to grep — but we're at /workspace/document, not temp
        result = layer.deduce(
            "search for budget analysis in the file",
            current_state="/workspace/document",
        )

        # Should detect mismatch since files were directed to temp
        assert result["mismatch_score"] > 0
        # If confidence is high enough, should emit cd prerequisite
        if result["prerequisites"]:
            assert "cd" in result["prerequisites"]
            assert result["target"] == "/workspace/document/temp"

    def test_domain_agnostic_auth_example(self, bare_layer):
        """Verify the layer works for non-filesystem domains."""
        bare_layer.add_transition(
            name="auth_required",
            directing_actions=["login", "authenticate", "connect"],
            operating_actions=["read", "write", "delete", "post"],
            prerequisite="authenticate",
        )

        # Observe: connected to a service
        bare_layer.observe(
            state="service_a",
            actions=["connect"],
            targets=["service_b"],
        )

        # Try to read from service_b while at service_a
        result = bare_layer.deduce("read the data", current_state="service_a")
        assert result["mismatch_score"] > 0

    def test_transition_dataclass(self):
        """Verify Transition strengthen/weaken mechanics."""
        vec = generate_symbol(89, "test", 10000)
        t = Transition(name="test", vector=vec, prerequisite="prep")
        assert t.strength == 1.0
        assert t.fire_count == 0

        t.strengthen(0.1)
        assert t.strength > 1.0
        assert t.fire_count == 1

        t.weaken(0.5)
        assert t.strength < 1.1  # weakened from strengthened value


# ── Packs ──

class TestPacks:
    def test_filesystem_pack_loads(self):
        layer = DeductiveLayer(dimension=10000, seed=89, packs=["filesystem"])
        assert "location_change" in layer.transitions
        t = layer.transitions["location_change"]
        assert t.prerequisite == "cd"

    def test_pack_deduction_works(self):
        layer = DeductiveLayer(dimension=10000, seed=89, packs=["filesystem"])
        layer.observe(state="/home", actions=[])
        layer.observe(state="/home", actions=["mv"], targets=["/tmp"])
        result = layer.deduce(query="grep for the file", current_state="/home")
        assert result["prerequisites"] == ["cd"]
        assert result["confidence"] > 0

    def test_unknown_pack_raises(self):
        with pytest.raises(FileNotFoundError):
            DeductiveLayer(packs=["nonexistent_domain"])


# ── Temporal Stagnation Detection ──


class TestTemporalStagnation:
    """Tests for Level 2 mismatch: state unchanged despite directing actions."""

    @pytest.fixture
    def temporal_layer(self):
        """DeductiveLayer with temporal tracking enabled."""
        return DeductiveLayer(
            dimension=10000, seed=89, packs=["filesystem"],
            temporal_window=3,
        )

    def test_stagnation_fires_after_directing_actions(self, temporal_layer):
        """State unchanged after mkdir + mv → stagnation detected → cd prerequisite."""
        # Initial state observation
        temporal_layer.observe(state="/home", actions=[])
        # Turn 1: mkdir — directing action, state unchanged
        temporal_layer.observe(state="/home", actions=["mkdir"], targets=["/home/temp"])
        # Turn 2: mv — directing action, state still unchanged
        temporal_layer.observe(state="/home", actions=["mv"], targets=["/home/temp"])

        # Query implies operating action — stagnation should trigger
        result = temporal_layer.deduce("grep for the file", current_state="/home")
        assert result["mismatch_score"] > 0.10
        # Should emit cd prerequisite
        assert result["prerequisites"] == ["cd"]
        assert result["confidence"] > 0

    def test_no_stagnation_when_state_changes(self, temporal_layer):
        """State changing normally → no stagnation signal."""
        temporal_layer.observe(state="/home", actions=[])
        temporal_layer.observe(state="/home/src", actions=["cd"])
        temporal_layer.observe(state="/home/src/test", actions=["cd"])

        result = temporal_layer.deduce("grep for the file", current_state="/home/src/test")
        # No stagnation — state has been changing
        stagnation = temporal_layer._compute_stagnation()
        assert stagnation < 0.5

    def test_no_stagnation_without_directing_actions(self, temporal_layer):
        """No directing actions → stagnation score should be 0."""
        temporal_layer.observe(state="/home", actions=[])
        temporal_layer.observe(state="/home", actions=["ls"])
        temporal_layer.observe(state="/home", actions=["cat"])

        stagnation = temporal_layer._compute_stagnation()
        assert stagnation == 0.0

    def test_stagnation_clears_after_resolve(self, temporal_layer):
        """After a resolving action (cd), stagnation should clear."""
        temporal_layer.observe(state="/home", actions=[])
        temporal_layer.observe(state="/home", actions=["mkdir"], targets=["/home/temp"])
        temporal_layer.observe(state="/home", actions=["mv"], targets=["/home/temp"])

        # Stagnation should be high before resolve
        assert temporal_layer._compute_stagnation() > 0.0

        # Resolving action (cd) clears stagnation
        temporal_layer.observe(state="/home/temp", actions=["cd"])
        assert temporal_layer._compute_stagnation() == 0.0

    def test_stagnation_one_shot(self, temporal_layer):
        """After stagnation fires, it should clear (one-shot deduction)."""
        temporal_layer.observe(state="/home", actions=[])
        temporal_layer.observe(state="/home", actions=["mkdir"], targets=["/home/temp"])
        temporal_layer.observe(state="/home", actions=["mv"], targets=["/home/temp"])

        # First deduce — should fire
        result1 = temporal_layer.deduce("grep for files", current_state="/home")
        assert result1["prerequisites"] == ["cd"]

        # Second deduce — should NOT fire (one-shot cleared state)
        result2 = temporal_layer.deduce("cat the output", current_state="/home")
        assert result2["prerequisites"] == []

    def test_temporal_window_respects_size(self):
        """Temporal window limits how far back stagnation looks."""
        layer = DeductiveLayer(
            dimension=10000, seed=89, packs=["filesystem"],
            temporal_window=2,
        )
        layer.observe(state="/home", actions=[])
        # Old directing action — outside window of 2
        layer.observe(state="/home", actions=["mkdir"], targets=["/home/temp"])
        # Two non-directing turns
        layer.observe(state="/home", actions=["ls"])
        layer.observe(state="/home", actions=["cat"])

        stagnation = layer._compute_stagnation()
        # Window=2 looks at last 2 turns (ls, cat) — no directing actions
        assert stagnation == 0.0

    def test_stagnation_domain_agnostic(self):
        """Stagnation works with any registered domain, not just filesystem."""
        layer = DeductiveLayer(dimension=10000, seed=89)
        layer.add_transition(
            name="auth_gate",
            directing_actions=["connect", "request"],
            operating_actions=["read", "write", "delete"],
            prerequisite="authenticate",
        )

        layer.observe(state="unauthenticated", actions=[])
        layer.observe(state="unauthenticated", actions=["connect"])
        layer.observe(state="unauthenticated", actions=["request"])

        result = layer.deduce("read the data", current_state="unauthenticated")
        assert result["mismatch_score"] > 0.10
        if result["prerequisites"]:
            assert "authenticate" in result["prerequisites"]


# ── Beam State Prediction ──


class TestBeamPrediction:
    """Tests for Level 3 mismatch: beam-predicted state divergence."""

    @pytest.fixture
    def beam_layer(self):
        """DeductiveLayer with beam prediction enabled."""
        return DeductiveLayer(
            dimension=10000, seed=89, packs=["filesystem"],
            temporal_window=3, enable_beam=True,
        )

    def test_beam_needs_minimum_history(self, beam_layer):
        """Beam prediction requires ≥3 turns of history."""
        beam_layer.observe(state="/home", actions=[])
        beam_layer.observe(state="/home", actions=["mkdir"])

        # Only 2 turns — beam should return 0
        div = beam_layer._compute_beam_divergence("/home")
        assert div == 0.0

    def test_beam_stores_state_glyphs(self, beam_layer):
        """Each observe() stores a minimal Glyph for beam prediction."""
        beam_layer.observe(state="/home", actions=[])
        beam_layer.observe(state="/home/src", actions=["cd"])
        beam_layer.observe(state="/home/src/test", actions=["cd"])

        assert len(beam_layer._state_glyph_history) == 3

    def test_beam_enabled_flag(self):
        """When enable_beam=False, no beam prediction infrastructure."""
        layer = DeductiveLayer(dimension=10000, seed=89, enable_beam=False)
        layer.observe(state="/home", actions=[])
        assert len(layer._state_glyph_history) == 0
        assert layer._beam_predictor is None

    def test_beam_reset_clears_history(self, beam_layer):
        """Reset clears beam glyph history."""
        beam_layer.observe(state="/home", actions=[])
        beam_layer.observe(state="/home/src", actions=["cd"])
        assert len(beam_layer._state_glyph_history) == 2

        beam_layer.reset()
        assert len(beam_layer._state_glyph_history) == 0

    def test_beam_divergence_with_stagnation(self, beam_layer):
        """Beam prediction with stagnant state should detect divergence."""
        # Build up some history with state changes
        beam_layer.observe(state="/home", actions=[])
        beam_layer.observe(state="/home", actions=["mkdir"], targets=["/home/temp"])
        beam_layer.observe(state="/home", actions=["mv"], targets=["/home/temp"])
        beam_layer.observe(state="/home", actions=["cp"], targets=["/home/temp"])

        # With 4 turns of history, beam prediction should be available
        assert len(beam_layer._state_glyph_history) == 4

        # Divergence calculation should not error
        div = beam_layer._compute_beam_divergence("/home")
        assert isinstance(div, float)
        assert 0.0 <= div <= 1.0
