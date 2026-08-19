# 2A-4 in-band view ablation (Day4, frozen C1 stem)

files=2  windows=4  smoke=True
Day5 unused. No training. E_all_inband is a control.

| Arm | iq | fft | amp_phase | stem rel-L2 | Reading |
| --- | --- | --- | --- | -----: | --- |
| R0_full | full | full | full | 0.0653 | LIVE |
| A_amp | full | full | inband | 0.0651 | LIVE |
| B_iq | inband | full | full | 0.0652 | LIVE |
| C_fft | full | inband | full | 0.0017 | STABLE |
| D_iq_amp | inband | full | inband | 0.0651 | LIVE |
| E_all_inband | inband | inband | inband | 0.0000 | STABLE |

smallest STABLE kill = C_fft
verdict = SMALLEST_KILL
STABLE < 0.01; LIVE ≥ 0.05. Do not train from this file.
