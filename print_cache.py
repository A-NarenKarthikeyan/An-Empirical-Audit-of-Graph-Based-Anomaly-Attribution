import json

with open('results/faithfulness_eval_cache.jsonl', 'r') as f:
    lines = f.readlines()

print(f"Total cache entries: {len(lines)}")
for i in [3, 4, 5]:
    if i < len(lines):
        data = json.loads(lines[i])
        print(f"\n--- ENTRY {i} ---")
        print(f"Key: {data.get('cache_key')}")
        resp = data.get('response', {})
        print(f"Timestamp: {resp.get('timestamp', 'NONE')}")
        print(f"Root causes: {resp.get('root_cause_sensors')}")
        print(f"Explanation: {resp.get('explanation')}")
