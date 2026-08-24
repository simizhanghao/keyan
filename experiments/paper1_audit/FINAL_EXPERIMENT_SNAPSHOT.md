# Final Experiment Snapshot

Status: **FROZEN BEFORE X6**. Official blind receivers were sealed when this snapshot was generated.

- Canonical checkpoints: `25`
- Development receivers/packets: `14 / 112000`
- Official blind receivers: `6 (SEALED)`
- Git parent: `837c7f495ba1dd455662a4f466ad97b4e35016ab`

## Canonical checkpoints

| Model | Seed | Epoch | Params | SHA256 |
|---|---:|---:|---:|---|
| Shen-CIS | 0 | 74 | 13510424 | `c29b4f7f29f0a259d2d13b393b90c39486856435db976a8e84018e0e52284bb1` |
| Shen-CIS | 1 | 74 | 13510424 | `783f6d2bc010648601afe02bc78a0376d070c48d74167268302a7af9945e24e4` |
| Shen-CIS | 2 | 74 | 13510424 | `8c3df50bbd88c2dd172b900217911389de21c35b9aec7513578f7218adda3baf` |
| Shen-CIS | 3 | 74 | 13510424 | `ee519bfb10ef20cf381afee1867818efdef537ece39ad54858847a94992816ca` |
| Shen-CIS | 4 | 74 | 13510424 | `bebfb811e298777492017ac375ecbd7e567b9fe76821c285bbb46209f119cfa7` |
| Shen-RA | 0 | 71 | 13510424 | `a420f4ec74c1c66d6f64159be4514b72166b6cb6c23ed10382ae1bd4930d8998` |
| Shen-RA | 1 | 71 | 13510424 | `8526e28f2e37c2df5a17e00cb8020df897636d8fcdbdd9e3f0c472268a2eff8b` |
| Shen-RA | 2 | 71 | 13510424 | `c3d81b848d2cc9cda675ed5b1676afa1b74606672696aad9d728584647f984ff` |
| Shen-RA | 3 | 71 | 13510424 | `9f2da5e12b651ecc9fe230369d38c41f18fae566d59865935ec15a9f5b14ccdc` |
| Shen-RA | 4 | 71 | 13510424 | `02e845a1eeb4154c656b004183e6d0625bfce57f477712a46a5b3d0dd0058f65` |
| B1-OOB | 0 | 5 | 539178 | `581579577b208ecacdc20011a15da0ad2ec280cc1b31fe1e1394c2b3b8216a99` |
| B1-OOB | 1 | 5 | 539178 | `44c6c031627cd111ebab1767bc0ff047f61d13669d44233421aec3e9b89f9f89` |
| B1-OOB | 2 | 5 | 539178 | `f90b3295ff516e8ff3bc0b48b37e36e391b0ea00bacbf51f46517f4eb0446482` |
| B1-OOB | 3 | 5 | 539178 | `99e0ad25787aebdd759bb1f4d66681d1e8262df97d3cbb8dca6497caa48e2ee7` |
| B1-OOB | 4 | 5 | 539178 | `cf3b767b914e08540a72b4a51a346e887eb953538e27bc849b0ef0e6382b8cc1` |
| C'-OOB | 0 | 4 | 625896 | `dfa675b856e1a61e46ad82a058fe1c9fa7f92aa10725718e7c581c83dc865b67` |
| C'-OOB | 1 | 4 | 625896 | `8a7d1d0a9671ea929dda08b526b0e88fb21226275697c3ae9e8191665ac66388` |
| C'-OOB | 2 | 4 | 625896 | `e9ae272dff3603edc473299de7fa9219a4baa65e14068636660fa35d794ef1b5` |
| C'-OOB | 3 | 4 | 625896 | `3a8b0b8657c1cdef84a0cfc0d80309beae21278d93fcecd5aea7cee78405f077` |
| C'-OOB | 4 | 4 | 625896 | `8b5d80a69e40d3e4860fddff36859d920280f1f71b6a2e44e9477e5e08ed3e2e` |
| C'-TrueIB | 0 | 5 | 580008 | `c4a1e5c5c777d161d555aab58d1d9fe0b7f5735a3c28df30654ed4be9be5e3b6` |
| C'-TrueIB | 1 | 5 | 580008 | `dd758f6dc65b034b6b02c9963324c7a79edb433409f1ec1d8fe84522ec8956d1` |
| C'-TrueIB | 2 | 5 | 580008 | `71e487ad6a8dcbce42a72ffe833ab73f55f64347917328d1f24c046f601a8794` |
| C'-TrueIB | 3 | 5 | 580008 | `a4ab63a9cc672d85f7322a5f4fe05b2ced8ce01211a3339f70a745d103910852` |
| C'-TrueIB | 4 | 5 | 580008 | `3746dac76fe49a62991f43fa06a5e590fd287016776c02e898ca55bb53d9c6b4` |

## Environment

- `python`: `3.12.12`
- `pytorch`: `2.10.0+cu129`
- `cuda_runtime`: `12.9`
- `numpy`: `2.4.4`
- `h5py`: `3.16.0`
- `platform`: `Linux-5.15.0-164-generic-x86_64-with-glibc2.35`
- `git_parent`: `837c7f495ba1dd455662a4f466ad97b4e35016ab`
- `gpu`: `NVIDIA A100-SXM4-80GB, 570.158.01`

## Code hashes

- `new_phase/experiments/paper1_audit/scripts/run_final_shen.py`: `4ff4d35a76a99e4e79e13a59c24d7c49fa472d9e3172bd716b845712b8a48812`
- `new_phase/experiments/paper1_audit/scripts/run_final_short.py`: `93b279221fd61fcc5f011ad65f8925243787c416dfa13fe39e114c2e9f533df1`
- `new_phase/experiments/paper1_audit/scripts/build_cis_cache.py`: `785de1d109e147c2de542953be14af3e66c45774f819c01727b437dc5a0db9ff`
- `new_phase/experiments/paper1_audit/scripts/run_x3_scale.py`: `f5234d2d584a4587c8f7e14a75a2831041e12f4b19849199a7e3788137d9190a`
- `new_phase/experiments/paper1_audit/scripts/run_x3_shuffle.py`: `a2303e1749abfe71467e1abf5da941be93f51ce46c5dff8771a1b8ed77e10a01`
- `new_phase/experiments/paper1_audit/scripts/run_x3_occlusion.py`: `b620658078d3d5315baf74a46d301694a57baafd3ed2ffe345d9c71bd5602ddf`
- `new_phase/experiments/paper1_audit/scripts/build_x6_neutral.py`: `05646418e5bf8667b0a7a3b7882e1d5b4906f1366d7c4a3eb2325c8143626d22`
- `new_phase/experiments/paper1_audit/scripts/run_x6_confirmatory.py`: `e91cc8294666bc1eb63cdc3cd3b94ee12a633a97e584394c65ba39514330792a`

## Frozen artifact hashes

- `new_phase/experiments/paper1_audit/FINAL_DATA_MANIFEST.md`: `054adc62095b1810ff601a1740b5d8b7ff3cee87d470be7a51cdaedc79e28222`
- `new_phase/experiments/paper1_audit/THREE_PAPER_BOUNDARY_LOCK.md`: `46138f0529f84fb20817051ef189142f82e772c3f879c3f0c018a96f7077a1d5`
- `new_phase/experiments/paper1_audit/PAPER_C_PREREG_LOCK.md`: `599d66d1a4e546564a9abdf13a873b554d1f30a92529faf2a00e2eda0ed42e57`
- `new_phase/experiments/paper1_audit/X6_PROTOCOL_LOCK.md`: `517b3f3eb3483884c5709b581907442428e4d4f8fbff95b1ce2a6f559d2d451f`
- `new_phase/experiments/paper1_audit/results/final_training/x6_neutral/development_neutral.npy`: `e4457a1592d148e93648f812907d614651c4545da1701e6077fa9b8a8261668a`
- `new_phase/experiments/paper1_audit/results/final_training/x6_neutral/development_neutral.json`: `956bb7534dfc2902e11482cd3c2ceffd0f7dd89e1ce22a816f06b6de6be0eb83`

## Blind archive

- SHA256: `e8f4cdc32cbbb7e6cfa410375e7412e204a2c412996d2a22a55ab0c8b9ff79f1`
- Signal values opened: `false`

## Excluded artifacts

- Mixed-batch Shen-CIS seeds 0-3 under `results/final_training/shen/`.
- No-checkpoint short runs under `results/final_training/short/`.

After first blind access, retraining, checkpoint reselection, hyperparameter tuning, preprocessing changes, intervention changes, and model additions are prohibited.
