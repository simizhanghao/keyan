# Author Block Patch Plan

## Current state (from compiled PDF / `main.tex`)

```latex
\author{Author~One, Author~Two%
\thanks{Manuscript prepared on branch \texttt{paper-ready-v3}. Affiliations TBD.}}
...
\section*{Acknowledgment}
TBD.
```

## Problems for submission

1. Placeholder author names visible to reviewers (single-blind journal).
2. Draft git-branch footnote is not acceptable in camera-ready / submission PDF.
3. `Acknowledgment TBD` reads as incomplete manuscript.

## Proposed replacement (template only — fill with real data)

```latex
\author{
First~Author, Second~Author, and Third~Author%
\thanks{First Author is with the Department of ..., University of ..., City, Country (e-mail: first.author@university.edu).}
\thanks{Second Author is with ...}
\thanks{Third Author is with ...}
\thanks{Corresponding author: First Author.}}
```

Adjust to IEEE IoTJ author-block conventions once final author count is known.

## Acknowledgment options

| Case | Action |
|------|--------|
| No funding | Remove `\section*{Acknowledgment}` entirely |
| Funding exists | Replace `TBD` with grant numbers / agency names |
| Code/data only | Optional short sentence in Acknowledgment or move link to Appendix only |

## Appendix A reproducibility

Current text points to `https://github.com/hanCChan/lunwen` branch `paper-ready-v3`.

Before submission, confirm:

- Public repo URL is final (lunwen vs rf-hstu-lora)
- Branch/tag name is stable for reviewers
- All coauthors consent to public code release

## This round

**No patch applied.** Waiting for user-provided author order and affiliations.
