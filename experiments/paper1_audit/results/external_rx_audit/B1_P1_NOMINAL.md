# B1 P1 nominal pilot

**Verdict: pipeline PASS; no paper metric.**

Fixed fold `P1 = rtl_2` and seed 0 were run with source-only checkpoint logic.
The pilot used 4 packets per HDF5 file and one epoch solely to verify real
multi-view forward/backward execution.

- 14 files; 52 source packets; 4 held-out packets
- parameters: `539,178`
- loss: `2.3082 -> 1.1991`
- source accuracy: `1.0` (pilot)
- held-out accuracy: `1.0` (pilot only, too few packets to interpret)
- official six blind receivers: unopened

Machine-readable result: `b1_p1_nominal_seed0.json`.
