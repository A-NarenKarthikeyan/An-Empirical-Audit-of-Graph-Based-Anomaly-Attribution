# NVIDIA NIM model availability and probe results

- Endpoint: `https://integrate.api.nvidia.com/v1`
- `GET /models`: HTTP 200; 84 model IDs saved in `models_list.txt`.
- The list is the endpoint response. Candidate existence in `/models` does not imply callable deployment: three selected candidates returned HTTP 404.
- Tiny call: temperature 0, max_tokens 30, user message `Reply with the word OK.`
- Realistic packet: first `packet_text` in `results_b/final/audit_blind.csv`; S6 v2 system prompt from `results_b/final/prompt_and_scorer.md`; temperature 0, max_tokens 600. Only the parser result and time are saved, not the model's realistic response.

## Keyword matches

- **llama** (15): meta/codellama-70b, meta/llama-3.2-11b-vision-instruct, meta/llama-3.2-90b-vision-instruct, meta/llama-guard-4-12b, meta/llama2-70b, nvidia/llama-3.1-nemoguard-8b-content-safety, nvidia/llama-3.1-nemoguard-8b-topic-control, nvidia/llama-3.1-nemotron-51b-instruct, nvidia/llama-3.1-nemotron-70b-instruct, nvidia/llama-3.1-nemotron-safety-guard-8b-v3, nvidia/llama-3.1-nemotron-ultra-253b-v1, nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1, nvidia/llama-3.2-nv-embedqa-1b-v1, nvidia/llama-nemotron-embed-vl-1b-v2, nvidia/llama3-chatqa-1.5-70b
- **mistral** (8): mistralai/codestral-22b-instruct-v0.1, mistralai/mistral-7b-instruct-v0.3, mistralai/mistral-large, mistralai/mistral-large-2-instruct, mistralai/mixtral-8x22b-v0.1, nv-mistralai/mistral-nemo-12b-instruct, nvidia/mistral-nemo-minitron-8b-8k-instruct, nvidia/nv-embedqa-mistral-7b-v2
- **mixtral** (1): mistralai/mixtral-8x22b-v0.1
- **qwen** (0): (none)
- **gemma** (8): google/codegemma-1.1-7b, google/codegemma-7b, google/diffusiongemma-26b-a4b-it, google/gemma-2b, google/gemma-3-12b-it, google/gemma-3-4b-it, google/gemma-4-31b-it, google/recurrentgemma-2b
- **phi** (2): microsoft/phi-3-vision-128k-instruct, microsoft/phi-3.5-moe-instruct
- **nemotron** (20): nvidia/llama-3.1-nemotron-51b-instruct, nvidia/llama-3.1-nemotron-70b-instruct, nvidia/llama-3.1-nemotron-safety-guard-8b-v3, nvidia/llama-3.1-nemotron-ultra-253b-v1, nvidia/llama-nemotron-embed-vl-1b-v2, nvidia/nemotron-3-embed-1b, nvidia/nemotron-3-nano-omni-30b-a3b-reasoning, nvidia/nemotron-3-super-120b-a12b, nvidia/nemotron-3-ultra-550b-a55b, nvidia/nemotron-3.5-bolt-e4b, nvidia/nemotron-3.5-content-safety, nvidia/nemotron-3.5-embed-1b, nvidia/nemotron-3.5-embed-8b, nvidia/nemotron-3.5-lightning-30b-a3b, nvidia/nemotron-3.5-super-vl-120b-a12b, nvidia/nemotron-4-340b-instruct, nvidia/nemotron-4-340b-reward, nvidia/nemotron-nano-3-30b-a3b, nvidia/nemotron-parse, nvidia/nemotron-parse-2.0
- **deepseek** (2): deepseek-ai/deepseek-coder-6.7b-instruct, deepseek-ai/deepseek-v4.1-flash

## Calls

| Model ID | Status | Tiny seconds | Tiny response | Realistic seconds | Parsable | Error |
|---|---:|---:|---|---:|---|---|
| mistralai/mistral-7b-instruct-v0.3 | 404 | 0.354100 | — | — | — | HTTP 404 Not Found: {"status":404,"title":"Not Found","detail":"Function '[request-id-redacted]': Not found for account '[account-id-redacted]'"} |
| meta/llama-3.2-11b-vision-instruct | 200 | 5.256080 | OK. | 23.519796 | yes | — |
| google/gemma-3-12b-it | 404 | 0.300490 | — | — | — | HTTP 404 Not Found: {"status":404,"title":"Not Found","detail":"Function '[request-id-redacted]': Not found for account '[account-id-redacted]'"} |
| microsoft/phi-3.5-moe-instruct | 404 | 0.291381 | — | — | — | HTTP 404 Not Found: {"status":404,"title":"Not Found","detail":"Function '[request-id-redacted]': Not found for account '[account-id-redacted]'"} |

## Estimated total time

Using the one successful realistic call as the per-call basis (23.519796 seconds/call):

- 164 calls: **3,857.246544 seconds** (estimate).
- 240 calls: **5,644.751040 seconds** (estimate).

These are extrapolations from a single call, not measured batch runtimes.
