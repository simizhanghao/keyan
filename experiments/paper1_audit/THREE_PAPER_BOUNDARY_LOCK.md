# Three-Paper Boundary Lock

Status: FROZEN BEFORE FIRST OFFICIAL BLIND ACCESS, 2026-08-24.

The three papers answer different scientific questions and use different
deployment assumptions. Evidence may be cited across papers only with explicit
disclosure; the same experiment cannot be presented as a headline contribution
in more than one paper.

| Scope | Paper A | Paper B | Paper C |
|---|---|---|---|
| Question | selective OOB fusion under same-RX cross-day shift | receiver-sensitive OOB shortcut mechanism under zero-target unseen RX | few-shot target-RX calibration |
| Primary data | OSU | 20-SDR | 20-SDR target enrollment |
| Target labels | 0 | 0 | K-shot > 0 |
| OSU Day-5 result | headline candidate | context only | excluded |
| C' architecture | applied model | probe backbone, not novelty | optional source encoder |
| OOB fusion ablation | Paper A only | not repeated as architecture claim | excluded |
| X1/X3 mechanism | excluded | headline evidence | motivation only |
| TrueIB | optional ordinary control | primary control | excluded |
| Shen-CIS/RA | excluded | strong baseline | optional source reference |
| RSA | excluded | negative boundary | excluded |
| Official blind X6 | excluded | confirmatory test | later target domains only |
| RCPA | excluded | excluded | candidate method |

Paper A is closed-set identification, not authentication, receiver-independent
RFFI, or a broad robustness claim. Paper B is a mechanism paper, not a proposed
architecture leaderboard. Paper C starts only after Paper B submission and only
if its preregistered method gate is met. Neither Paper A nor Paper B will be
submitted to IEEE Internet of Things Journal without prior written permission.
