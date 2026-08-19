# RX source-only manifest audit

verdict=RX_MANIFEST_PASS  training=false  gpu=false  oracle=false

| direction | train RX | val RX | test RX | devices | missing | ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rx1_to_rx2 | [1] | [1] | [2] | 24 | 0 | True |
| rx2_to_rx1 | [2] | [2] | [1] | 24 | 0 | True |

Day4 C'/F0 checkpoints must not be loaded onto these manifests.
Oracle target-val manifests are forbidden.
