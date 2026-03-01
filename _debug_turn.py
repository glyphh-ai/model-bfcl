"""Debug script: trace what the LLM sees and returns for turn 0."""
import sys, json
sys.path.insert(0, ".")

from handler import GlyphhBFCLHandler
from llm_client import get_client
from multi_turn_handler import get_available_functions, extract_turn_query, _extract_fs_hints

handler = GlyphhBFCLHandler(threshold=0.15)
llm = get_client("anthropic", "claude-haiku-4-5-20251001")

with open("data/bfcl/BFCL_v4_multi_turn_base.json") as f:
    entry = json.loads(f.readline())

funcs = get_available_functions(entry)
query = extract_turn_query(entry["question"][0])
print(f"Query: {query}")

route = handler.route(query, funcs)
scores = route.get("all_scores", [])
print(f"Top 5 scores: {[(s['function'], round(s['score'],3)) for s in scores[:5]]}")

# Reconstruct the prompt
func_map = {fd["name"]: fd for fd in funcs}
score_map = {s["function"]: s["score"] for s in scores}
top_names = sorted(func_map.keys(), key=lambda n: score_map.get(n, 0.0), reverse=True)

func_blocks = []
for fname in top_names:
    fdef = func_map[fname]
    desc = fdef.get("description", "")
    params = fdef.get("parameters", {}).get("properties", {})
    required = fdef.get("parameters", {}).get("required", [])
    score = score_map.get(fname, 0.0)
    param_parts = [f"{p}: {params[p].get('type','any')} ({'required' if p in required else 'optional'})" for p in params]
    func_blocks.append(f"  [{score:.2f}] {fname}({', '.join(param_parts)}) -- {desc}")

functions_text = "AVAILABLE FUNCTIONS:\n" + "\n".join(func_blocks)

system = (
    "You select the correct function call(s) for each turn of a stateful conversation. "
    "Given the user's request, the current system state, and the available functions "
    "(ranked by semantic relevance), call exactly what the user is asking for.\n\n"
    'Return ONLY a JSON array. Each element has ONE key = function name, value = arguments object. '
    'Example: [{"cd": {"folder": "src"}}, {"mv": {"source": "old.txt", "destination": "new.txt"}}]. '
    "Return [] if no function matches. No explanation, no markdown."
)

user_msg = f"{functions_text}\n\nUser: {query}"

messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]

print(f"\n--- PROMPT ({len(user_msg)} chars) ---")
print(user_msg[:500])
print("...")

# Raw call
try:
    text = llm.chat_complete(messages, max_tokens=512)
    print(f"\n--- RAW LLM RESPONSE ---")
    print(repr(text))

    text_stripped = text.strip()
    if text_stripped and not text_stripped.startswith("["):
        text_stripped = "[" + text_stripped + "]"
    calls = json.loads(text_stripped)
    print(f"\nParsed calls: {calls}")

    valid_names = {fd["name"] for fd in funcs}
    result = {}
    for call in calls:
        if isinstance(call, dict):
            for fname, args in call.items():
                if fname in valid_names:
                    result[fname] = args if isinstance(args, dict) else {}
    print(f"Valid result: {result}")
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
