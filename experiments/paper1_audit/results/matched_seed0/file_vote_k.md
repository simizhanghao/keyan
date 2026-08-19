# Paper 1 file-vote K sensitivity (Day4, frozen 1C)

verdict=H4_PASS  sanity_ok=True  day5=unused
prefix=first K by window_index  primary=mean_logits

## File-Acc % mean±std (5 seeds)

### mean_logits

| Model           |      K=8 |     K=16 |     K=32 |     K=64 |    K=128 |    K=256 |
| --------------- | -------: | -------: | -------: | -------: | -------: | -------: |
| B Main          | 21.7±23.1 | 21.7±21.5 | 22.5±24.4 | 24.2±27.5 | 21.7±25.1 | 23.3±27.3 |
| C' Full ratio   | 65.8±5.4 | 66.7±8.8 | 66.7±6.6 | 73.3±6.3 | 75.8±6.8 | 79.2±8.3 |
| A CNN           | 56.7±8.6 | 61.7±9.0 | 64.2±6.3 | 65.8±8.5 | 65.8±8.5 | 65.8±8.5 |

### mean_prob

| Model           |      K=8 |     K=16 |     K=32 |     K=64 |    K=128 |    K=256 |
| --------------- | -------: | -------: | -------: | -------: | -------: | -------: |
| B Main          | 19.2±19.0 | 20.8±21.2 | 21.7±26.1 | 22.5±27.7 | 20.8±24.5 | 22.5±26.6 |
| C' Full ratio   | 61.7±5.4 | 69.2±8.1 | 67.5±6.2 | 73.3±7.6 | 75.0±5.1 | 76.7±5.6 |
| A CNN           | 56.7±7.6 | 60.8±7.6 | 62.5±7.2 | 62.5±8.3 | 63.3±7.5 | 62.5±5.9 |

### majority

| Model           |      K=8 |     K=16 |     K=32 |     K=64 |    K=128 |    K=256 |
| --------------- | -------: | -------: | -------: | -------: | -------: | -------: |
| B Main          | 17.5±20.5 | 19.2±21.4 | 22.5±25.8 | 19.2±22.4 | 20.0±22.9 | 21.7±26.1 |
| C' Full ratio   | 64.2±6.3 | 70.8±7.2 | 72.5±7.0 | 72.5±8.6 | 72.5±6.3 | 75.8±6.8 |
| A CNN           | 58.3±2.9 | 58.3±5.9 | 57.5±4.6 | 59.2±5.4 | 59.2±5.4 | 59.2±5.4 |

## Pre-registered H4

H4a smooth (C' mean_logits, max step drop ≤ 2.0 pp): True  curve=[65.83, 66.67, 66.67, 73.33, 75.83, 79.17]
H4b C' > CNN at K=64 mean_logits: True
SPIKE_ONLY (lose all K≤64, win only at 256): False

Frozen 1C claim table is still K=256 / mean_logits. This does not open Day5, LODO, or RX2.
