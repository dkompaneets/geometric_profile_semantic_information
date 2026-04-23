# A Geometric Profile of Semantic Information in Text

**Dmitriy Kompaneets** (Independent Researcher) — `dkompaneets@gmail.com`

Paper: [`geometric_profile_semantic_information.pdf`](geometric_profile_semantic_information.pdf)
Walkthrough: [`explainer.md`](explainer.md) · [`explainer.pdf`](explainer.pdf)

## TL;DR

Shannon entropy measures uncertainty over symbols and is indifferent to meaning; pairwise metrics like BERTScore compare two texts rather than characterizing one. This work measures the **semantic content of a single text** via the geometry of its sentence embeddings: a three-coordinate profile `P_E(T) = (N, B, I)` for novelty, breadth, and integration. Within a fixed embedding, six natural axioms uniquely determine a scalar measure; a **no-go theorem** then shows that no scalar summary of the profile can simultaneously be analytically stable, ordinally robust, and cross-representation comparable. Two practical scalars (`S_minmax`, `S_rank`) occupy distinct corners of this trade-off triangle. A separate variational result equates the breadth coordinate to the log-determinant of a determinantal point process (ρ = 0.985 on 507 Project Gutenberg chapters), giving breadth an optimization-theoretic foundation.

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook validation.ipynb
```

Run top to bottom. First run downloads external datasets (STS-B, SICK, SummEval, XSum, EUR-Lex, arXiv abstracts) into `./.cache/`. Subsequent runs are fast. Project Gutenberg texts are pre-cached in `gutenberg_cache/`.

Every numeric claim in the paper is produced by this single notebook.

## File map

| File | Purpose |
|---|---|
| `geometric_profile_semantic_information.tex` / `.pdf` | Paper source + compiled PDF |
| `validation.ipynb` | Reproducibility — produces every number in the paper |
| `explainer.md` / `.pdf` | Self-contained technical walkthrough of all concepts |
| `unicode_map.tex` | LaTeX header for rebuilding `explainer.pdf` via `pandoc + tectonic` |
| `requirements.txt` | Python dependencies |
| `gutenberg_cache/` | Pre-downloaded novel texts (Project Gutenberg) |
| `LICENSE` | MIT |

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```
@misc{kompaneets2026geometric,
  author  = {Kompaneets, Dmitriy},
  title   = {A Geometric Profile of Semantic Information in Text:
             Frame-Conditional Uniqueness and a Trade-Off Triangle for Scalar Summaries},
  year    = {2026},
  note    = {arXiv preprint},
  url     = {https://github.com/dkompaneets/geometric_profile_semantic_information}
}
```
