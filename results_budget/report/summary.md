# Experiment summary

## KV capacity (FP16 vs FP8)
| workload | context_len | kv_dtype | kv_blocks | max_concurrency_analytic | first_pressure_C |
|---|---|---|---|---|---|
| corpus | 2048 | auto | 1.32e+04 | 92 | 16 |
| corpus | 2048 | fp8 | 2.61e+04 | 181 | 4 |
| shared_prefix | 2048 | auto | 1.36e+04 | 94 | 32 |
| shared_prefix | 2048 | fp8 | 2.69e+04 | 186 | 32 |
| synthetic | 512 | auto | 1.32e+04 | 276 | 16 |
| synthetic | 512 | fp8 | 2.61e+04 | 543 | 16 |
| synthetic | 2048 | auto | 1.32e+04 | 92 | 16 |
| synthetic | 2048 | fp8 | 2.61e+04 | 181 | 4 |
| synthetic | 7680 | auto | 1.32e+04 | 26 | 4 |
| synthetic | 7680 | fp8 | 2.61e+04 | 52 | 4 |

### Capacity gain
| workload | context_len | analytic_base | analytic_other | analytic_gain_x | pressure_C_base | pressure_C_other |
|---|---|---|---|---|---|---|
| corpus | 2048 | 92 | 181 | 1.97 | 16 | 4 |
| shared_prefix | 2048 | 94 | 186 | 1.98 | 32 | 32 |
| synthetic | 512 | 276 | 543 | 1.97 | 16 | 16 |
| synthetic | 2048 | 92 | 181 | 1.97 | 16 | 4 |
| synthetic | 7680 | 26 | 52 | 2 | 4 | 4 |

## Throughput knee (turns concurrency into throughput, or hits a compute wall?)
| workload | context_len | kv_dtype | peak_tps | knee_C |
|---|---|---|---|---|
| corpus | 2048 | auto | 860 | 64 |
| shared_prefix | 2048 | auto | 1.26e+03 | 256 |
| synthetic | 2048 | auto | 857 | 64 |
| synthetic | 512 | auto | 2.84e+03 | 256 |
| synthetic | 7680 | auto | 221 | 16 |
| corpus | 2048 | fp8 | 1.14e+03 | 256 |
| shared_prefix | 2048 | fp8 | 1.91e+03 | 256 |
| synthetic | 2048 | fp8 | 1.14e+03 | 256 |
| synthetic | 512 | fp8 | 3.41e+03 | 256 |
| synthetic | 7680 | fp8 | 302 | 64 |

## Speculative decoding crossover C*
| workload | kv_dtype | spec | context_len | speedup_at_min_C | speedup_at_max_C | c_star | c_star_tps | status | mean_acceptance_rate |
|---|---|---|---|---|---|---|---|---|---|
| corpus | auto | draft_model | 2048 | 1.51 | 1.44 | - | 24.5 | no_crossover_in_range | 0.525 |
| corpus | auto | eagle | 2048 | 1.01 | 0.974 | 16.6 | 8.02 | crosses | 0.161 |
| corpus | auto | ngram | 2048 | 1.09 | 0.829 | 25.9 | 10.3 | crosses | 0.114 |
| corpus | fp8 | draft_model | 2048 | 1.34 | 2.15 | - | 25.1 | no_crossover_in_range | 0.528 |
| corpus | fp8 | eagle | 2048 | 1.08 | 1.32 | 43.6 | 13.4 | crosses | 0.166 |
| corpus | fp8 | ngram | 2048 | 1.22 | 1.23 | - | 15 | no_crossover_in_range | 0.121 |

## Quality: FP8 vs FP16 KV cache
| context_len | ppl_auto | ppl_fp8 | delta | delta_pct |
|---|---|---|---|---|
| 512 | 10.8 | 10.9 | 0.0191 | 0.176 |
| 2048 | 8.24 | 8.26 | 0.0217 | 0.264 |
| 4096 | 7.69 | 7.7 | 0.0176 | 0.229 |
