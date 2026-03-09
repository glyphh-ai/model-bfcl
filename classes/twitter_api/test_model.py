"""Tests for the TwitterAPI Glyphh Ada model.

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


# -- Intent extraction tests -----------------------------------------------

class TestIntentExtraction:
    """Verify NL queries extract correct Twitter actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # authenticate
        ("Log in to Twitter with username 'john' and password 'john1234'", "authenticate"),
        ("Authenticate using my username 'dr_smith' and password is securePass123", "authenticate"),
        ("If you need to log in, here's my username 'john' and password 'john1234'", "authenticate"),
        ("my password, i forgot to tell you, is 'john1234'", "authenticate"),
        # post_tweet
        ("Draft a tweet about my recent travel escapades", "post_tweet"),
        ("Could you post a tweet saying 'Excited for the trip!'", "post_tweet"),
        ("Craft a tweet that states 'Managed to archive important data files!'", "post_tweet"),
        ("Toss a tweet out there about this comparative analysis", "post_tweet"),
        ("Let's share the updates on tire status", "post_tweet"),
        ("Help me maintain a social media presence by crafting a tweet", "post_tweet"),
        # comment
        ("Add a comment saying 'Great job!'", "comment"),
        ("Drop a comment underneath the tweet", "comment"),
        ("Comment on that tweet something like 'Is this pressure too low?'", "comment"),
        ("Append a thoughtful comment to the tweet", "comment"),
        ("Use a supportive comment 'Cheers!'", "comment"),
        # retweet
        ("Retweet the tweet I just posted", "retweet"),
        ("Could you assist in retweeting that fantastic post", "retweet"),
        ("Can you amplify its reach by retweeting it?", "retweet"),
        ("Give it a boost by retweeting it for me", "retweet"),
        ("Widen the circle of those who might share", "retweet"),
        # mention
        ("Add a mention of @RoadsideAssistance to the tweet", "mention"),
        ("Let's ensure it gains more momentum by tagging @technewsworld", "mention"),
        # follow / unfollow
        ("Follow user @someone", "follow"),
        ("Unfollow user @spammer", "unfollow"),
        # search_tweets
        ("Search for tweets about climate change", "search_tweets"),
        # get_user_stats
        ("Get user stats for john", "get_user_stats"),
        # get_login_status
        ("Check my login status", "get_login_status"),
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
            assert func.startswith("TwitterAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every TwitterAPI function has a reverse action mapping."""
        expected_funcs = {
            "authenticate_twitter", "post_tweet", "comment", "retweet",
            "mention", "follow_user", "unfollow_user", "get_tweet",
            "get_tweet_comments", "get_user_tweets", "get_user_stats",
            "list_all_following", "search_tweets", "posting_get_login_status",
        }
        for func in expected_funcs:
            assert func in FUNC_TO_ACTION, f"Missing FUNC_TO_ACTION for {func}"
            assert func in FUNC_TO_TARGET, f"Missing FUNC_TO_TARGET for {func}"


# -- HDC encoding tests ----------------------------------------------------

class TestEncoding:
    """Verify Glyphs encode correctly and score as expected."""

    @pytest.fixture
    def encoder(self):
        return Encoder(ENCODER_CONFIG)

    @pytest.fixture
    def func_defs(self):
        """Load actual function definitions from func_doc."""
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "posting_api.json"
        with open(path) as f:
            defs = [json.loads(line) for line in f if line.strip()]
        # Prefix with TwitterAPI class
        for fd in defs:
            bare = fd["name"]
            fd["name"] = f"TwitterAPI.{bare}"
        return defs

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
        """All 14 functions should be encoded."""
        assert len(func_glyphs) == 14

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Log in with username john and password abc", "TwitterAPI.authenticate_twitter"),
        ("Post a tweet saying 'Hello world!'", "TwitterAPI.post_tweet"),
        ("Add a comment saying 'Great work!'", "TwitterAPI.comment"),
        ("Retweet the tweet I just posted", "TwitterAPI.retweet"),
        ("Add a mention of @friend to the tweet", "TwitterAPI.mention"),
        ("Follow user @newconnection", "TwitterAPI.follow_user"),
        ("Unfollow user @spammer", "TwitterAPI.unfollow_user"),
        ("Retrieve a specific tweet by ID", "TwitterAPI.get_tweet"),
        ("Get the comments on this tweet", "TwitterAPI.get_tweet_comments"),
        ("Get all tweets from user john", "TwitterAPI.get_user_tweets"),
        ("Show user stats for my profile", "TwitterAPI.get_user_stats"),
        ("List all users that I am following", "TwitterAPI.list_all_following"),
        ("Search for tweets about machine learning", "TwitterAPI.search_tweets"),
        ("Check my login status on the platform", "TwitterAPI.posting_get_login_status"),
    ])
    def test_function_routing(self, encoder, func_glyphs, query: str, expected_func: str):
        """Each function should be the top match for its representative query."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        top_score = scores[0][1]

        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func} (score={top_score:.4f})\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    @pytest.mark.parametrize("query, expected_func", [
        # Multi-turn context queries (from actual BFCL test entries)
        ("Help me maintain a social media presence by crafting a tweet that states, 'Managed to archive important data files!' using the hashtags #DataManagement and #Efficiency", "TwitterAPI.post_tweet"),
        ("Once the tweet is live, reinforce the achievement by commenting underneath with a phrase like 'Another successful task completed today!'", "TwitterAPI.comment"),
        ("Wonderful! I'd really appreciate it if you could assist in retweeting that fantastic post about tire maintenance", "TwitterAPI.retweet"),
        ("After posting, make sure to add a mention of @RoadsideAssistance to the tweet we just created", "TwitterAPI.mention"),
        ("If you need to log in, here's my username 'john' and password 'john1234'", "TwitterAPI.authenticate_twitter"),
        ("Retweet the tweet I just posted to widen its reach within my network", "TwitterAPI.retweet"),
        ("The comment content should be 'Another successful task completed today!'", "TwitterAPI.comment"),
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
            "Post a tweet about travel",
            "Retweet my latest post",
            "Comment on the tweet with encouragement",
            "Search for tweets about technology",
            "Log in with my credentials",
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
