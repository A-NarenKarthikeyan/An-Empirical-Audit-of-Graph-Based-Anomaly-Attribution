import json

cache_path = 'results/faithfulness_eval_cache.jsonl'
inflate_cases = []
with open(cache_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        if d['cache_key'].endswith('_value_pert'):
            resp = d['response']
            expl = resp.get('explanation', '')
            if '8.50' in expl: # It successfully reasoned about 8.50 std
                inflate_cases.append(d)

print(f"Found {len(inflate_cases)} inflate cases. Printing 4...")
for i, d in enumerate(inflate_cases[:4]):
    print(f"\n====================== CASE {i+1} ======================")
    print(f"KEY: {d['cache_key']}")
    print(f"SENSORS: {d['response']['root_cause_sensors']}")
    print(f"EXPLANATION:\n{d['response']['explanation']}")
