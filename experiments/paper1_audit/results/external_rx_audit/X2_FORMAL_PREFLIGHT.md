# X2 formal pilot preflight

**Verdict: PASS.** No signal values from official blind receivers were opened.

- source inventory: 14 receivers, each 8,000 packets;
- pseudo-unseen folds: `rtl_2`, `rtl_5`, `b200_1`, `b200_mini_1`, `b210_1`, `pluto_1`;
- seeds: 0 and 1;
- models: B1 and audited C';
- total runs: 24;
- optimizer: AdamW, `lr=1e-3`, `wd=5e-4`;
- checkpoint selection: source validation only;
- official six blind receivers: sealed.

Machine-readable run manifest: `x2_formal_manifest.json`.
