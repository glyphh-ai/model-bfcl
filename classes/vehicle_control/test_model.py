"""Tests for the VehicleControlAPI Glyphh Ada model.

Verifies:
  1. Intent extraction: NL queries -> correct action/target
  2. HDC encoding: queries and functions encode into compatible Glyphs
  3. Routing accuracy: each function is the top match for its representative queries
"""

import json
import sys
from pathlib import Path

import pytest

# Setup path -- local dir MUST come first to shadow top-level intent.py
_DIR = Path(__file__).parent
_BFCL_DIR = _DIR.parent.parent
sys.path.insert(0, str(_DIR))

from intent import extract_intent, ACTION_TO_FUNC, FUNC_TO_ACTION, FUNC_TO_TARGET
from encoder import ENCODER_CONFIG, encode_query, encode_function

from glyphh import Encoder
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity


def _layer_weighted_score(q: Glyph, f: Glyph) -> float:
    """Layer-weighted similarity using ENCODER_CONFIG weights.

    Computes per-layer scores (weighted-average of role similarities within
    each layer), then combines layers using their similarity_weight.
    This lets the intent layer's action role dominate over noisy BoW roles.
    """
    config = ENCODER_CONFIG
    layer_configs = {lc.name: lc for lc in config.layers}

    total_score = 0.0
    total_weight = 0.0

    for lname in q.layers:
        if lname not in f.layers:
            continue
        lc = layer_configs.get(lname)
        if not lc:
            continue
        lw = lc.similarity_weight

        # Weighted average of role similarities within this layer
        role_score = 0.0
        role_weight = 0.0
        for seg_cfg in lc.segments:
            sname = seg_cfg.name
            if sname not in q.layers[lname].segments or sname not in f.layers[lname].segments:
                continue
            for role_cfg in seg_cfg.roles:
                rname = role_cfg.name
                qs = q.layers[lname].segments[sname]
                fs = f.layers[lname].segments[sname]
                if rname in qs.roles and rname in fs.roles:
                    sim = float(cosine_similarity(qs.roles[rname].data, fs.roles[rname].data))
                    rw = role_cfg.similarity_weight
                    role_score += sim * rw
                    role_weight += rw

        if role_weight > 0:
            layer_sim = role_score / role_weight
            total_score += layer_sim * lw
            total_weight += lw

    return total_score / total_weight if total_weight > 0 else 0.0


# -- Intent extraction tests ------------------------------------------------

class TestIntentExtraction:
    """Verify NL queries extract correct vehicle control actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # parking brake
        ("Engage the parking brake before getting out", "parking_brake"),
        ("Set the parking brake as a safety measure", "parking_brake"),
        ("Make sure the handbrake is engaged", "parking_brake"),
        # start engine
        ("Start the engine with ignition mode START", "start_engine"),
        ("Fire up the engine for the trip", "start_engine"),
        ("Initiate the engine using START mode", "start_engine"),
        # climate control
        ("Adjust the climate control to 72 degrees", "climate_control"),
        ("Set the temperature to 25 celsius", "climate_control"),
        ("Turn on the air conditioning", "climate_control"),
        # lock doors
        ("Lock all doors before driving", "lock_doors"),
        ("Secure the doors of the vehicle", "lock_doors"),
        ("Unlock the doors so passengers can exit", "lock_doors"),
        # fill fuel
        ("Fill the fuel tank with 30 gallons", "fill_fuel"),
        ("Refuel the vehicle before the trip", "fill_fuel"),
        ("Add 20 gallons of gasoline to the tank", "fill_fuel"),
        # check tires
        ("Check the tire pressure on all wheels", "check_tires"),
        ("Inspect the tires before the road trip", "check_tires"),
        # find tire shop
        ("Find the nearest tire shop", "find_tire_shop"),
        ("Where is the closest tire shop?", "find_tire_shop"),
        # headlights
        ("Turn on the headlights", "headlights"),
        ("Set the headlights to auto mode", "headlights"),
        # cruise control
        ("Set cruise control to 65 mph", "cruise_control"),
        ("Activate cruise control at 70 mph", "cruise_control"),
        # navigation
        ("Navigate to 123 Main Street", "navigation"),
        ("Set the destination to downtown", "navigation"),
        # estimate distance
        ("Estimate the distance between two cities", "estimate_distance"),
        ("How far is it between New York and Boston?", "estimate_distance"),
        # drive feasibility
        ("Can I drive 200 miles on the current fuel?", "drive_feasibility"),
        # car status
        ("Display the status of the fuel level", "car_status"),
        ("Check the fuel level of the vehicle", "car_status"),
        # display log
        ("Display the log messages", "display_log"),
        ("Show the log of recent actions", "display_log"),
        # get speed
        ("What is the current speed of the vehicle?", "get_speed"),
        ("How fast am I going?", "get_speed"),
        # outside temperature
        ("Get the outside temperature", "temp_google"),
        # get zipcode
        ("What is the zipcode for San Francisco?", "get_zipcode"),
        # gallon to liter
        ("Convert 10 gallons to liters", "gallon_to_liter"),
        # liter to gallon
        ("Convert 20 liters to gallons", "liter_to_gallon"),
        # press brake
        ("Press the brake pedal fully", "press_brake"),
        ("Apply the brake to slow down", "press_brake"),
        # release brake
        ("Release the brake pedal", "release_brake"),
    ])
    def test_action_extraction(self, query: str, expected_action: str):
        result = extract_intent(query)
        assert result["action"] == expected_action, (
            f"Query: '{query}'\n"
            f"Expected action: {expected_action}\n"
            f"Got: {result['action']}"
        )

    def test_all_actions_have_functions(self):
        """Every action in the lexicon maps to a real function."""
        for action, func in ACTION_TO_FUNC.items():
            if action == "none":
                continue
            assert func.startswith("VehicleControlAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every VehicleControlAPI function has a reverse action mapping."""
        expected_funcs = {
            "activateParkingBrake", "adjustClimateControl", "check_tire_pressure",
            "displayCarStatus", "display_log", "estimate_distance",
            "estimate_drive_feasibility_by_mileage", "fillFuelTank",
            "find_nearest_tire_shop", "gallon_to_liter", "get_current_speed",
            "get_outside_temperature_from_google",
            "get_outside_temperature_from_weather_com",
            "get_zipcode_based_on_city", "liter_to_gallon", "lockDoors",
            "pressBrakePedal", "releaseBrakePedal", "setCruiseControl",
            "setHeadlights", "set_navigation", "startEngine",
        }
        for func in expected_funcs:
            assert func in FUNC_TO_ACTION, f"Missing FUNC_TO_ACTION for {func}"
            assert func in FUNC_TO_TARGET, f"Missing FUNC_TO_TARGET for {func}"


# -- HDC encoding tests -----------------------------------------------------

class TestEncoding:
    """Verify Glyphs encode correctly and score as expected."""

    @pytest.fixture
    def encoder(self):
        return Encoder(ENCODER_CONFIG)

    @pytest.fixture
    def func_defs(self):
        """Load actual function definitions from func_doc."""
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "vehicle_control.json"
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @pytest.fixture
    def func_glyphs(self, encoder, func_defs):
        """Encode all function defs into Glyphs."""
        glyphs = {}
        for fd in func_defs:
            cd = encode_function(fd)
            glyph = encoder.encode(Concept(name=cd["name"], attributes=cd["attributes"]))
            glyphs[fd["name"]] = glyph
        return glyphs

    def _score(self, encoder, func_glyphs, query: str) -> list[tuple[str, float]]:
        """Score a query against all function Glyphs using hierarchical scoring."""
        qd = encode_query(query)
        q_glyph = encoder.encode(Concept(name=qd["name"], attributes=qd["attributes"]))

        scores = []
        for fname, fg in func_glyphs.items():
            sim = _layer_weighted_score(q_glyph, fg)
            scores.append((fname, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def test_all_functions_encoded(self, func_glyphs):
        """All 22 functions should be encoded."""
        assert len(func_glyphs) == 22

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Engage the parking brake on the vehicle", "activateParkingBrake"),
        ("Adjust the climate control to 72 degrees fahrenheit", "adjustClimateControl"),
        ("Check the tire pressure on all four tires", "check_tire_pressure"),
        ("Display the fuel level status of the vehicle", "displayCarStatus"),
        ("Display the log messages from the system", "display_log"),
        ("Estimate the distance between two city zipcodes", "estimate_distance"),
        ("Can I drive 200 miles with the current mileage and fuel?", "estimate_drive_feasibility_by_mileage"),
        ("Fill the fuel tank with 30 gallons of gas", "fillFuelTank"),
        ("Find the nearest tire shop for a replacement", "find_nearest_tire_shop"),
        ("Convert 10 gallons to liters", "gallon_to_liter"),
        ("What is the current speed of the vehicle?", "get_current_speed"),
        ("Get the outside temperature from Google", "get_outside_temperature_from_google"),
        ("Get the outside temperature from weather.com", "get_outside_temperature_from_weather_com"),
        ("What is the zipcode for San Francisco?", "get_zipcode_based_on_city"),
        ("Convert 20 liters to gallons", "liter_to_gallon"),
        ("Lock all doors of the vehicle securely", "lockDoors"),
        ("Press the brake pedal to slow the vehicle", "pressBrakePedal"),
        ("Release the brake pedal to start moving", "releaseBrakePedal"),
        ("Set cruise control to 65 miles per hour", "setCruiseControl"),
        ("Turn on the headlights for night driving", "setHeadlights"),
        ("Navigate to 123 Main Street downtown", "set_navigation"),
        ("Start the engine with ignition mode START", "startEngine"),
    ])
    def test_function_routing(self, encoder, func_glyphs, query: str, expected_func: str):
        """Each function should be the top match for its representative query."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        top_score = scores[0][1]
        second_score = scores[1][1] if len(scores) > 1 else 0.0

        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func} (score={top_score:.4f})\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    @pytest.mark.parametrize("query, expected_func", [
        # Multi-turn context queries (from actual BFCL entries)
        ("Make sure the car is ready for the adventure, doors locked and handbrake engaged", "activateParkingBrake"),
        ("Start up the engine for the road trip ahead", "startEngine"),
        ("Fill the tank with 30 gallons before we head out", "fillFuelTank"),
        ("Secure all doors of the vehicle before driving", "lockDoors"),
        ("Set the cabin temperature to a comfortable 25 degrees celsius", "adjustClimateControl"),
        ("Check if tire pressure is safe to proceed on the highway", "check_tire_pressure"),
        ("Set navigation to 456 Oak Avenue, Pinehaven, IL", "set_navigation"),
    ])
    def test_multi_turn_queries(self, encoder, func_glyphs, query: str, expected_func: str):
        """Queries from actual multi-turn entries should route correctly."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func}\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    def test_separation(self, encoder, func_glyphs):
        """Top match should have meaningful separation from second match."""
        test_queries = [
            "Start the engine of the car",
            "Lock all the doors",
            "Fill the fuel tank with gasoline",
            "Check the tire pressure",
            "Navigate to the destination",
        ]
        for query in test_queries:
            scores = self._score(encoder, func_glyphs, query)
            top = scores[0][1]
            second = scores[1][1]
            gap = top - second
            assert gap > 0.01, (
                f"Query: '{query}' -- insufficient separation\n"
                f"Top: {scores[0][0]}={top:.4f}, Second: {scores[1][0]}={second:.4f}, Gap={gap:.4f}"
            )
