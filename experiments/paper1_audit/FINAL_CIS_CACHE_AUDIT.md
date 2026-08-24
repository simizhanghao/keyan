# Final CIS Cache Audit

Date: 2026-08-24

The cache contains only the 14 development receiver files: 112,000 packets in
total, exactly 8,000 packets per receiver. The official six blind receivers
were not read.

Final training has 112,000 samples and 1,750 optimizer steps per epoch at the
locked batch size 64. An X4-C fold has 104,000 samples and 1,625 steps; the
final dataset is therefore only 7.7% larger, not an order of magnitude larger.

A deterministic audit sampled eight packets from each development receiver
(112 total). Cached and online CIS tensors had identical shape `1 x 52 x 126`,
identical device labels, and exact float32 equality:

```text
max_abs_error = 0.0
bad_files = 0 / 14
raw_count = cache_count = 112000
```

The cache changes only data delivery. Batch size, model, loss, optimizer,
learning rate, seed set, and fixed epoch budgets remain unchanged.
