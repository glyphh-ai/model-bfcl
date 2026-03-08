"""
BFCL handler — HDC routing + argument extraction.

BFCLHandler ties together:
  BFCLScorer        — HDC function routing (which function to call)
  Extractor         — argument extraction (rule-based or LLM-assisted)
  CognitiveLoop     — multi-turn routing with episodic memory + deduction

Exports:
  BFCLHandler.route(query, func_defs)            → single-function routing result
  BFCLHandler.route_multi(query, func_defs)      → multi-function routing result
  BFCLHandler.route_multi_turn(entry, func_defs) → multi-turn routing via CognitiveLoop
  BFCLHandler.is_irrelevant(query, func_defs)    → irrelevance detection
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from glyphh import CognitiveLoop

from scorer import BFCLScorer
from extractor import ArgumentExtractor
from domain_config import CLASS_DOMAIN_CONFIGS
from intent import extract_api_class, extract_pack_actions

# Per-class exemplar directories
_CLASSES_DIR = Path(__file__).parent / "classes"

# Class name → folder name
_CLASS_TO_FOLDER = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI":        "twitter_api",
    "MessageAPI":        "message_api",
    "PostingAPI":        "posting_api",
    "TicketAPI":         "ticket_api",
    "MathAPI":           "math_api",
    "TradingBot":        "trading_bot",
    "TravelAPI":         "travel_booking",
    "TravelBookingAPI":  "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Class name → SDK pack names for CognitiveLoop
_CLASS_TO_PACKS: dict[str, list[str]] = {
    "GorillaFileSystem": ["filesystem"],
    "TwitterAPI":        ["social"],
    "MessageAPI":        ["messaging"],
    "PostingAPI":        ["social"],
    "TicketAPI":         ["ticket"],
    "MathAPI":           ["math"],
    "TradingBot":        ["trading"],
    "TravelAPI":         ["travel"],
    "TravelBookingAPI":  ["travel"],
    "VehicleControlAPI": ["vehicle"],
}

# Pack canonical action → class-prefixed function name (overrides for non-obvious mappings)
# For GorillaFileSystem, pack canonicals ARE bare function names — auto-derived.
# For other classes, explicit mappings where pack canonical != bare function name.
_PACK_FUNC_OVERRIDES: dict[str, dict[str, str]] = {
    "TwitterAPI": {
        "post":    "TwitterAPI.post_tweet",
        "repost":  "TwitterAPI.retweet",
        "comment": "TwitterAPI.comment",
        "mention": "TwitterAPI.mention",
        "follow":  "TwitterAPI.follow_user",
        "unfollow": "TwitterAPI.unfollow_user",
        "authenticate_social": "TwitterAPI.authenticate_twitter",
        "get_tweet": "TwitterAPI.get_tweet",
        "get_tweet_comments": "TwitterAPI.get_tweet_comments",
        "get_user_stats": "TwitterAPI.get_user_stats",
        "get_user_tweets": "TwitterAPI.get_user_tweets",
        "list_all_following": "TwitterAPI.list_all_following",
        "search_tweets": "TwitterAPI.search_tweets",
        "social_get_login_status": "TwitterAPI.posting_get_login_status",
    },
    "MessageAPI": {
        "dm_social": "MessageAPI.send_message",
        "send_message": "MessageAPI.send_message",
        "view_messages": "MessageAPI.view_messages_sent",
        "delete_message": "MessageAPI.delete_message",
        "search_messages": "MessageAPI.search_messages",
        "add_contact": "MessageAPI.add_contact",
        "get_user_id": "MessageAPI.get_user_id",
        "list_users": "MessageAPI.list_users",
        "get_message_stats": "MessageAPI.get_message_stats",
        "message_login": "MessageAPI.message_login",
        "message_get_login_status": "MessageAPI.message_get_login_status",
    },
    "PostingAPI": {
        "post":    "PostingAPI.post",
        "comment": "PostingAPI.comment",
        "repost":  "PostingAPI.share",
        "follow":  "PostingAPI.follow_user",
        "unfollow": "PostingAPI.unfollow_user",
        "authenticate_social": "PostingAPI.authenticate_twitter",
        "get_tweet": "PostingAPI.get_tweet",
        "get_tweet_comments": "PostingAPI.get_tweet_comments",
        "get_user_stats": "PostingAPI.get_user_stats",
        "get_user_tweets": "PostingAPI.get_user_tweets",
        "list_all_following": "PostingAPI.list_all_following",
        "search_tweets": "PostingAPI.search_tweets",
        "social_get_login_status": "PostingAPI.posting_get_login_status",
    },
    "MathAPI": {
        "add":        "MathAPI.add",
        "subtract":   "MathAPI.subtract",
        "multiply":   "MathAPI.multiply",
        "divide":     "MathAPI.divide",
        "power":      "MathAPI.power",
        "sqrt":       "MathAPI.square_root",
        "log":        "MathAPI.logarithm",
        "statistics": "MathAPI.mean",
        "mean":       "MathAPI.mean",
        "percentage": "MathAPI.percentage",
        "absolute_value": "MathAPI.absolute_value",
        "max_value":  "MathAPI.max_value",
        "min_value":  "MathAPI.min_value",
        "round_number": "MathAPI.round_number",
        "standard_deviation": "MathAPI.standard_deviation",
        "sum_values": "MathAPI.sum_values",
        "convert_units": "MathAPI.si_unit_conversion",
        "si_unit_conversion": "MathAPI.si_unit_conversion",
        "imperial_si_conversion": "MathAPI.imperial_si_conversion",
    },
    "TradingBot": {
        "buy":           "TradingBot.buy",
        "sell":          "TradingBot.sell",
        "get_quote":     "TradingBot.get_quote",
        "get_balance":   "TradingBot.get_balance",
        "get_history":   "TradingBot.get_history",
        "place_order":   "TradingBot.place_order",
        "cancel_order":  "TradingBot.cancel_order",
        "add_to_watchlist": "TradingBot.add_to_watchlist",
        "remove_from_watchlist": "TradingBot.remove_stock_from_watchlist",
        "get_watchlist":  "TradingBot.get_watchlist",
        "filter_stocks_by_price": "TradingBot.filter_stocks_by_price",
        "fund_account":   "TradingBot.fund_account",
        "withdraw_funds": "TradingBot.withdraw_funds",
        "get_account_info": "TradingBot.get_account_info",
        "get_available_stocks": "TradingBot.get_available_stocks",
        "get_current_time": "TradingBot.get_current_time",
        "get_order_details": "TradingBot.get_order_details",
        "get_order_history": "TradingBot.get_order_history",
        "get_stock_info": "TradingBot.get_stock_info",
        "get_symbol_by_name": "TradingBot.get_symbol_by_name",
        "get_transaction_history": "TradingBot.get_transaction_history",
        "notify_price_change": "TradingBot.notify_price_change",
        "trading_login":  "TradingBot.trading_login",
        "trading_logout": "TradingBot.trading_logout",
        "trading_get_login_status": "TradingBot.trading_get_login_status",
    },
    "TravelBookingAPI": {
        "book_flight":      "TravelBookingAPI.book_flight",
        "book_hotel":       "TravelBookingAPI.book_hotel",
        "cancel_booking":   "TravelBookingAPI.cancel_booking",
        "check_in":         "TravelBookingAPI.check_in",
        "get_flight_status": "TravelBookingAPI.get_flight_status",
        "authenticate_travel": "TravelBookingAPI.authenticate_travel",
        "compute_exchange_rate": "TravelBookingAPI.compute_exchange_rate",
        "contact_customer_support": "TravelBookingAPI.contact_customer_support",
        "get_all_credit_cards": "TravelBookingAPI.get_all_credit_cards",
        "get_booking_history": "TravelBookingAPI.get_booking_history",
        "get_budget_fiscal_year": "TravelBookingAPI.get_budget_fiscal_year",
        "get_credit_card_balance": "TravelBookingAPI.get_credit_card_balance",
        "get_flight_cost":  "TravelBookingAPI.get_flight_cost",
        "get_nearest_airport": "TravelBookingAPI.get_nearest_airport_by_city",
        "list_all_airports": "TravelBookingAPI.list_all_airports",
        "purchase_insurance": "TravelBookingAPI.purchase_insurance",
        "register_credit_card": "TravelBookingAPI.register_credit_card",
        "retrieve_invoice":  "TravelBookingAPI.retrieve_invoice",
        "set_budget_limit":  "TravelBookingAPI.set_budget_limit",
        "travel_get_login_status": "TravelBookingAPI.travel_get_login_status",
        "verify_traveler_info": "TravelBookingAPI.verify_traveler_information",
    },
    "TravelAPI": {
        "book_flight":      "TravelAPI.book_flight",
        "book_hotel":       "TravelAPI.book_hotel",
        "cancel_booking":   "TravelAPI.cancel_booking",
        "check_in":         "TravelAPI.check_in",
        "get_flight_status": "TravelAPI.get_flight_status",
        "authenticate_travel": "TravelAPI.authenticate_travel",
        "compute_exchange_rate": "TravelAPI.compute_exchange_rate",
        "contact_customer_support": "TravelAPI.contact_customer_support",
        "get_all_credit_cards": "TravelAPI.get_all_credit_cards",
        "get_booking_history": "TravelAPI.get_booking_history",
        "get_budget_fiscal_year": "TravelAPI.get_budget_fiscal_year",
        "get_credit_card_balance": "TravelAPI.get_credit_card_balance",
        "get_flight_cost":  "TravelAPI.get_flight_cost",
        "get_nearest_airport": "TravelAPI.get_nearest_airport_by_city",
        "list_all_airports": "TravelAPI.list_all_airports",
        "purchase_insurance": "TravelAPI.purchase_insurance",
        "register_credit_card": "TravelAPI.register_credit_card",
        "retrieve_invoice":  "TravelAPI.retrieve_invoice",
        "set_budget_limit":  "TravelAPI.set_budget_limit",
        "travel_get_login_status": "TravelAPI.travel_get_login_status",
        "verify_traveler_info": "TravelAPI.verify_traveler_information",
    },
    "VehicleControlAPI": {
        "accelerate":     "VehicleControlAPI.accelerate",
        "brake":          "VehicleControlAPI.brake",
        "lock":           "VehicleControlAPI.lockDoors",
        "unlock":         "VehicleControlAPI.lockDoors",
        "start_engine":   "VehicleControlAPI.startEngine",
        "stop_engine":    "VehicleControlAPI.startEngine",
        "set_climate":    "VehicleControlAPI.adjustClimateControl",
        "set_navigation": "VehicleControlAPI.set_navigation",
        "set_cruise":     "VehicleControlAPI.setCruiseControl",
        "set_headlights": "VehicleControlAPI.setHeadlights",
        "fill_fuel_tank": "VehicleControlAPI.fillFuelTank",
        "check_tire_pressure": "VehicleControlAPI.check_tire_pressure",
        "gallon_to_liter": "VehicleControlAPI.gallon_to_liter",
        "liter_to_gallon": "VehicleControlAPI.liter_to_gallon",
        "get_zipcode":    "VehicleControlAPI.get_zipcode_based_on_city",
        "estimate_distance": "VehicleControlAPI.estimate_distance",
        "estimate_drive_feasibility": "VehicleControlAPI.estimate_drive_feasibility_by_mileage",
        "display_car_status": "VehicleControlAPI.displayCarStatus",
        "display_log":    "VehicleControlAPI.display_log",
        "find_tire_shop": "VehicleControlAPI.find_nearest_tire_shop",
        "get_current_speed": "VehicleControlAPI.get_current_speed",
        "get_outside_temperature": "VehicleControlAPI.get_outside_temperature_from_google",
        "press_brake_pedal": "VehicleControlAPI.pressBrakePedal",
        "release_brake_pedal": "VehicleControlAPI.releaseBrakePedal",
        "activate_parking_brake": "VehicleControlAPI.activateParkingBrake",
        "check_fuel":     "VehicleControlAPI.displayCarStatus",
    },
    "TicketAPI": {
        "create_ticket":  "TicketAPI.create_ticket",
        "resolve_ticket": "TicketAPI.resolve_ticket",
        "get_ticket":     "TicketAPI.get_ticket",
        "close_ticket":   "TicketAPI.close_ticket",
        "edit_ticket":    "TicketAPI.edit_ticket",
        "get_user_tickets": "TicketAPI.get_user_tickets",
        "ticket_login":   "TicketAPI.ticket_login",
        "ticket_logout":  "TicketAPI.logout",
        "ticket_get_login_status": "TicketAPI.ticket_get_login_status",
    },
}


class BFCLHandler:
    """HDC routing + pluggable argument extraction.

    Usage:
        handler = BFCLHandler()                          # rule-based extraction
        handler = BFCLHandler(extractor=LLMExtractor())  # LLM-assisted extraction
        result  = handler.route(query, func_defs)
        # result["tool"]       — predicted function name (or None if irrelevant)
        # result["args"]       — extracted argument dict
        # result["confidence"] — HDC similarity score
        # result["top_k"]      — top-5 scored functions for debugging
    """

    def __init__(
        self,
        confidence_threshold: float = 0.22,
        irrelevance_threshold: float = 0.60,
        extractor: Any | None = None,
    ) -> None:
        self._scorer    = BFCLScorer()
        self._extractor = extractor or ArgumentExtractor()
        self._threshold = confidence_threshold
        self._irr_threshold = irrelevance_threshold
        self.language   = "python"  # Set per-category by runner
        self._scorer_cache: dict[str, BFCLScorer] = {}  # cls → pre-built scorer

    def _extract_args(self, query: str, func_def: dict) -> dict:
        """Call extractor with language awareness."""
        if hasattr(self._extractor, 'extract') and 'language' in self._extractor.extract.__code__.co_varnames:
            return self._extractor.extract(query, func_def, language=self.language)
        return self._extractor.extract(query, func_def)

    # ── Public API ───────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        func_defs: list[dict],
        force: bool = False,
    ) -> dict[str, Any]:
        """Route a query to the single best matching function.

        Args:
            query:     Natural language query.
            func_defs: List of available function definitions.
            force:     If True, always return the best match without threshold
                       check. Use for routing categories where a correct function
                       is guaranteed to exist. If False, applies the confidence
                       threshold (use for irrelevance detection categories).

        Returns a result dict with:
          tool         (str | None) — predicted function name, None if irrelevant
          args         (dict)       — extracted argument values
          confidence   (float)      — HDC similarity score [0, 1]
          is_irrelevant (bool)      — True when no function matches
          latency_ms   (float)      — wall-clock time in milliseconds
          top_k        (list)       — top-5 [{function, score}] for inspection
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score(query)

        elapsed = (time.perf_counter() - t0) * 1000

        if force:
            # Routing category: always pick best — a correct function is guaranteed
            irrelevant = False
            func_name  = result.functions[0] if result.functions else None
        else:
            irrelevant = result.is_irrelevant or result.confidence < self._threshold
            func_name  = result.functions[0] if result.functions and not irrelevant else None
        args: dict = {}

        if func_name:
            func_def = self._find_def(func_defs, func_name)
            if func_def:
                args = self._extract_args(query, func_def)

        return {
            "tool":          func_name,
            "args":          args,
            "confidence":    result.confidence,
            "is_irrelevant": irrelevant,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }

    def route_multi(self, query: str, func_defs: list[dict], force: bool = False) -> dict[str, Any]:
        """Route a query to potentially multiple functions.

        Returns a result dict with:
          tools        (list[str])  — selected function names (empty if irrelevant)
          args         (dict)       — {func_name: {param: value, ...}}
          confidence   (float)      — average HDC similarity of selected functions
          is_irrelevant (bool)      — True when no function matches
          latency_ms   (float)      — wall-clock time in milliseconds
          top_k        (list)       — top-5 [{function, score}] for inspection
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score_multi(query)

        elapsed = (time.perf_counter() - t0) * 1000

        if not force and result.is_irrelevant:
            return {
                "tools":         [],
                "args":          {},
                "confidence":    result.confidence,
                "is_irrelevant": True,
                "latency_ms":    elapsed,
                "top_k":         result.all_scores[:5],
            }

        all_args: dict[str, dict] = {}
        for fname in result.functions:
            func_def = self._find_def(func_defs, fname)
            if func_def:
                all_args[fname] = self._extract_args(query, func_def)

        return {
            "tools":         result.functions,
            "args":          all_args,
            "confidence":    result.confidence,
            "is_irrelevant": False,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }

    def is_irrelevant(self, query: str, func_defs: list[dict]) -> tuple[bool, dict]:
        """Determine whether a query is irrelevant (no function should be called).

        Uses the higher irrelevance_threshold for stricter filtering.

        Returns:
            (is_irrelevant: bool, route_result: dict)
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score(query)

        elapsed = (time.perf_counter() - t0) * 1000

        irrelevant = result.is_irrelevant or result.confidence < self._irr_threshold
        func_name = result.functions[0] if result.functions and not irrelevant else None
        args: dict = {}

        if func_name:
            func_def = self._find_def(func_defs, func_name)
            if func_def:
                args = self._extract_args(query, func_def)

        route_result = {
            "tool":          func_name,
            "args":          args,
            "confidence":    result.confidence,
            "is_irrelevant": irrelevant,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }
        return route_result["is_irrelevant"], route_result

    def setup_multi_turn(
        self,
        entry: dict,
        func_defs: list[dict],
    ) -> dict[str, Any]:
        """Initialize state for multi-turn routing.

        Creates CognitiveLoops per class for state tracking across turns.
        Call route_turn() for each turn, then update_state() after arg extraction.

        Returns context dict with loops, scorers, class_available, etc.
        """
        involved_classes = entry.get("involved_classes", [])
        initial_config = entry.get("initial_config", {})

        # Group func_defs by class prefix
        class_funcs: dict[str, list[dict]] = {}
        for f in func_defs:
            cls = f["name"].split(".")[0]
            class_funcs.setdefault(cls, []).append(f)

        class_available: dict[str, set[str]] = {
            cls: {f["name"] for f in funcs}
            for cls, funcs in class_funcs.items()
        }

        # Build per-class CognitiveLoops and scorers (cached across entries)
        loops: dict[str, CognitiveLoop] = {}
        scorers: dict[str, BFCLScorer] = {}
        for cls in class_funcs:
            if cls in self._scorer_cache:
                scorer = self._scorer_cache[cls]
            else:
                scorer = BFCLScorer()
                exemplars = self._load_exemplars(cls)
                if exemplars:
                    scorer.configure_from_exemplars(exemplars)
                else:
                    scorer.configure(class_funcs[cls])
                self._scorer_cache[cls] = scorer
            scorers[cls] = scorer

            packs = _CLASS_TO_PACKS.get(cls, [])
            domain_config = CLASS_DOMAIN_CONFIGS.get(cls)
            if domain_config:
                loop = CognitiveLoop(
                    packs=packs if packs else None,
                    domain_config=domain_config,
                    model_scorer=scorer,
                )
                init_state = self._build_initial_state(cls, initial_config)
                loop.begin(functions=class_funcs[cls], initial_state=init_state)
                loops[cls] = loop

        return {
            "involved_classes": involved_classes,
            "class_funcs": class_funcs,
            "class_available": class_available,
            "loops": loops,
            "scorers": scorers,
        }

    def route_turn(
        self,
        query: str,
        ctx: dict[str, Any],
    ) -> list[str]:
        """Route a single turn using multi-stage pipeline with CognitiveLoop deduction.

        Pipeline:
          1. Detect API class
          2. DomainConfig multi_action_keywords (multi-function detection)
          3. Pack canonical routing (secondary)
          4. HDC scorer fallback
          5. Exclusion rules
          6. CognitiveLoop deduction (state-aware prerequisite injection)

        State must be updated after arg extraction via update_turn_state().

        Returns list of class-prefixed function names.
        """
        involved_classes = ctx["involved_classes"]
        class_available = ctx["class_available"]
        scorers = ctx["scorers"]
        loops = ctx["loops"]

        # Stage 1: Detect API class
        matched_cls = extract_api_class(query, involved_classes)
        available = class_available.get(matched_cls, set())

        # Stage 2: DomainConfig multi_action_keywords (PRIMARY)
        domain_config = CLASS_DOMAIN_CONFIGS.get(matched_cls)
        pack_funcs: list[str] = []
        if domain_config and domain_config.multi_action_keywords:
            query_lower = query.lower()
            for fname, patterns in domain_config.multi_action_keywords.items():
                if fname not in available:
                    continue
                for pat in patterns:
                    if pat in query_lower:
                        if fname not in pack_funcs:
                            pack_funcs.append(fname)
                        break

        # Stage 3: Pack canonical routing (secondary)
        pack_canonicals = extract_pack_actions(query)
        overrides = _PACK_FUNC_OVERRIDES.get(matched_cls, {})
        for canonical in pack_canonicals:
            if canonical in overrides:
                fname = overrides[canonical]
            else:
                fname = f"{matched_cls}.{canonical}"
            if fname in available and fname not in pack_funcs:
                pack_funcs.append(fname)

        # Stage 4: HDC scorer fallback (use score_multi for multi-function)
        if not pack_funcs:
            scorer = scorers.get(matched_cls)
            if scorer:
                result = scorer.score_multi(query)
                if result.functions:
                    pack_funcs = list(result.functions)

        # Stage 5: Exclusion rules
        if domain_config and domain_config.exclusion_rules:
            to_remove: set[str] = set()
            for specific, generics in domain_config.exclusion_rules.items():
                if specific in pack_funcs:
                    for g in generics:
                        if g in pack_funcs:
                            to_remove.add(g)
            pack_funcs = [f for f in pack_funcs if f not in to_remove]

        # Stage 6: CognitiveLoop deduction with state tracking
        loop = loops.get(matched_cls)
        if loop:
            current_state = loop._state.get("primary", "")
            deduction = loop.deductive.deduce(
                query=query, current_state=current_state,
            )
            if deduction.get("prerequisites"):
                # Resolve prerequisite action names through action_to_func mapping
                action_map = domain_config.action_to_func if domain_config else {}
                for prereq in deduction["prerequisites"]:
                    prereq_func = action_map.get(prereq, f"{matched_cls}.{prereq}")
                    if prereq_func in available and prereq_func not in pack_funcs:
                        pack_funcs.insert(0, prereq_func)

        return pack_funcs

    @staticmethod
    def update_turn_state(
        ctx: dict[str, Any],
        calls: list[dict],
    ) -> None:
        """Update CognitiveLoop state after arg extraction.

        Feed the extracted function calls (with args) back to the loop
        so state_effects fire (e.g. cd updates CWD). Also observe()
        actions in the deductive layer so it can track directing vs
        operating action history for prerequisite injection.
        """
        loops = ctx["loops"]
        # Collect actions per class for deductive observation
        class_actions: dict[str, list[str]] = {}
        for call in calls:
            for fname, args in call.items():
                if not isinstance(args, dict):
                    continue
                for cls, loop in loops.items():
                    prefixed = f"{cls}.{fname}"
                    if prefixed in (loop._available_funcs or {}):
                        loop._update_state([{prefixed: args}])
                        class_actions.setdefault(cls, []).append(fname)
                        break

        # Feed actions to deductive layer for temporal tracking
        for cls, actions in class_actions.items():
            loop = loops[cls]
            current_state = loop._state.get("primary", "")
            loop.deductive.observe(
                state=current_state,
                actions=actions,
            )

    def route_multi_turn(
        self,
        entry: dict,
        func_defs: list[dict],
    ) -> dict[str, Any]:
        """Multi-turn routing (legacy interface, routes all turns at once).

        For state-aware routing, use setup_multi_turn() + route_turn() instead.
        """
        t0 = time.perf_counter()
        ctx = self.setup_multi_turn(entry, func_defs)
        turns = entry.get("question", [])
        per_turn: list[dict] = []

        for turn_messages in turns:
            if not turn_messages:
                per_turn.append({"functions": [], "confidence": 0.0})
                continue
            query = self._extract_turn_query(turn_messages)
            if not query:
                per_turn.append({"functions": [], "confidence": 0.0})
                continue
            funcs = self.route_turn(query, ctx)
            per_turn.append({
                "functions": funcs,
                "confidence": 0.8 if funcs else 0.0,
            })

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "per_turn": per_turn,
            "latency_ms": elapsed,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_exemplars(cls: str) -> list[dict]:
        """Load pre-built exemplars for a class from classes/{folder}/exemplars.jsonl."""
        folder = _CLASS_TO_FOLDER.get(cls)
        if not folder:
            return []
        path = _CLASSES_DIR / folder / "exemplars.jsonl"
        if not path.exists():
            return []
        exemplars = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    exemplars.append(json.loads(line))
        return exemplars

    @staticmethod
    def _build_initial_state(cls: str, initial_config: dict) -> dict:
        """Convert BFCL initial_config to CognitiveLoop state format.

        For GorillaFileSystem: extracts root directory name as primary state.
        For other classes: returns minimal state.
        """
        cls_config = initial_config.get(cls, {})

        if cls == "GorillaFileSystem" and "root" in cls_config:
            root = cls_config["root"]
            # Root is {dirname: {contents}} — extract the root dir name
            root_name = list(root.keys())[0] if root else ""
            return {
                "primary": root_name,
                "collections": {},
            }

        return {
            "primary": "",
            "collections": {},
        }

    @staticmethod
    def _to_loop_functions(func_defs: list[dict]) -> list[dict]:
        """Convert func_defs to CognitiveLoop format."""
        return [
            {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            }
            for f in func_defs
        ]

    @staticmethod
    def _extract_turn_query(turn_messages: list) -> str:
        """Extract user query from a turn's message list."""
        for msg in reversed(turn_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if turn_messages:
            last = turn_messages[-1]
            return last.get("content", "") if isinstance(last, dict) else ""
        return ""

    @staticmethod
    def _find_def(func_defs: list[dict], name: str | None) -> dict | None:
        if not name:
            return None
        return next((f for f in func_defs if f.get("name") == name), None)
