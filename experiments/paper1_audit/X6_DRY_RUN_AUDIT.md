# X6 Development Dry-Run Audit

Date: 2026-08-24. Official blind receiver signals remained sealed.

- Canonical checkpoint strict-load and clean inference passed for all five model families at seed 0.
- All eight non-clean intervention paths passed for B1-OOB and C'-OOB: four scale values, shuffle, neutral replacement, left scale, and right scale.
- Each path evaluated the same first 32 packets of development receiver `b200_1`; these values are runtime checks and are not paper results.
- Every output contained finite Accuracy, Macro-F1, confusion matrix, packet indices, labels, predictions, checkpoint hash, and protocol metadata.
- The `dev-dry-run` phase guard rejected a blind receiver filename before attempting file access.
- Total successful output files: 21 (five clean paths, plus sixteen intervention paths).

Decision: **DRY RUN PASS**. The runner and analysis schema are eligible for the final pre-blind freeze.
