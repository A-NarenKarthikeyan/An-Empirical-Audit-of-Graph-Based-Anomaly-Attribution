# Automated judge audit

Judge model: `meta/llama-3.2-90b-vision-instruct`

Judge outputs are compared with the frozen S6 v2 individual-honesty strict check (every stated sensor/magnitude supported by the packet). Non-JSON or invalid-schema replies are retained and counted, not silently dropped.

| Explainer model | Judge supported | Judge exaggerated | Supported vs strict agreement | Cohen's κ | Non-JSON/invalid | Empty content | Strict-scorer paired n |
|---|---:|---:|---:|---:|---:|---:|---:|
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning | 78/82 = 0.9512 [0.8812, 0.9809] | 51/82 = 0.6220 [0.5138, 0.7192] | 0.9125 | -0.0448 | 0 | 0 | 80 |
| meta/llama-3.2-11b-vision-instruct | 79/82 = 0.9634 [0.8979, 0.9875] | 31/82 = 0.3780 [0.2808, 0.4862] | 0.9634 | 0.0000 | 0 | 0 | 82 |

Total source items: 164; non-JSON/invalid-schema replies: 0; empty-content replies: 0; HTTP/transport failures or not-yet-run items: 0.

