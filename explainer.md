# Explainer: A Geometric Profile of Semantic Information in Text

A self-contained, technical-but-gentle walkthrough of the paper. Every symbol, every concept, defined before it is used. Written to be read in order, from the top.

---

## 1. The problem the paper solves

**Question.** Given a piece of text `T`, how much "meaning" does it carry — and how is that meaning structured?

**Why existing tools don't answer it.**

- **Shannon entropy** measures uncertainty over symbols (characters, tokens). By design, it is **indifferent to meaning**. A random shuffle of words has roughly the same entropy as coherent prose.
- **Pairwise similarity metrics** (BERTScore, BLEURT, ROUGE) compare *two* texts. They can't characterize the richness of *one* text on its own.
- **Philosophical accounts** (Bar-Hillel & Carnap, Dretske, Floridi) define semantic information via logic or situation theory — rigorous, but not computable on real text.

**The paper's approach.** Map each sentence of `T` to a high-dimensional vector using a neural sentence-embedding model. The **geometry of the resulting point cloud** — where the points sit, how they spread, how they connect — encodes the text's semantic structure. Measure meaning by measuring geometry.

---

## 2. Primitives

Every term the paper uses, defined from the ground up.

### 2.1 Sentence embedding

A **sentence-embedding model** `E` is a neural network that maps a piece of text `T` to a vector `E(T) ∈ ℝⁿ`. Typical dimension `n = 768` (for `all-mpnet-base-v2`, the paper's primary model). Outputs are ℓ₂-normalized (unit length). The design goal of `E`: semantically similar sentences get nearby vectors; dissimilar sentences get distant vectors.

### 2.2 Segments

A text `T` is split into segments — typically sentences — giving `T = (T₁, T₂, …, T_k)`. Each segment gets embedded: `e_i = E(T_i) ∈ ℝⁿ`. The raw embedding cloud is `X_T = {e₁, …, e_k}`.

- **`k`** = number of segments in the text.
- **`n`** = embedding dimension (e.g., 768).

### 2.3 Mean and covariance

From the cloud `X_T`:
- **`μ_T ∈ ℝⁿ`** — the mean vector (centroid): where the text sits in embedding space.
- **`C_T`** — the covariance matrix of the segments around `μ_T`. An `n × n` symmetric positive-semidefinite matrix describing how the segments scatter.

### 2.4 Neutral baseline

A **baseline corpus** of semantically generic sentences (e.g., "The meeting is scheduled for tomorrow afternoon."). Embedding these gives:
- **`μ_0 ∈ ℝⁿ`** — mean of generic discourse; the "semantic origin."
- **`Σ_0`** — covariance of generic discourse; the "natural spread" of ordinary English.

Displacement of a text's centroid from `μ_0` measures how far the text is from generic prose.

### 2.5 Mahalanobis distance

Ordinary Euclidean distance treats all directions equally. **Mahalanobis distance** is direction-aware: given covariance `Σ_0`, the Mahalanobis norm of a vector `v` is

$$\|v\|_M = \sqrt{v^\top \Sigma_0^{-1} v}.$$

Intuition: directions in which the baseline varies widely (common drift) are **down-weighted**; directions in which it's narrow (unusual for generic prose) are **up-weighted**. Moving one inch into a direction generic English never occupies counts more than one inch along a direction it wobbles in all the time.

### 2.6 Ledoit–Wolf shrinkage

In 768-dimensional space with a finite baseline corpus, the sample covariance `Σ̂_0` is near-singular — inverting it gives garbage. **Ledoit–Wolf shrinkage** mixes the sample covariance with a scaled identity matrix `λI`:

$$\Sigma_{\text{LW}} = (1 - \alpha) \cdot \hat{\Sigma}_0 + \alpha \cdot \lambda I,$$

with `α` chosen analytically to minimize expected squared error. The result is a well-conditioned estimate that can be inverted safely. The paper uses this whenever `Σ_0⁻¹` is needed.

### 2.7 Rank of a covariance

**`rank(C)`** of an `n × n` matrix = the number of nonzero eigenvalues, i.e., the dimension of the subspace that `C` actually occupies. If all segments lie on a line, `rank(C) = 1`. If they span a plane, `rank = 2`. Geometrically: the **number of independent directions** of variation.

### 2.8 Deduplication and the threshold `τ`

To prevent exact or near-duplicates from inflating measurements, the framework runs **agglomerative clustering** on `X_T` with cosine distance, merging any two segments whose cosine similarity exceeds a threshold `τ ∈ (0, 1)`. Each cluster is replaced by its (renormalized) centroid. The result is a **deduplicated set**:

$$\tilde{T} = \{c_1, \ldots, c_m\}, \quad m \leq k.$$

- **High `τ` (e.g., 0.90)** — strict. Only near-identical sentences merge. Fine resolution.
- **Low `τ` (e.g., 0.70)** — lenient. Topically related sentences merge. Coarse resolution. Default for natural prose.

### 2.9 Effective rank (smooth rank)

Raw `rank(C)` is brittle: in high-dimensional embeddings, noise makes nearly every eigenvalue strictly nonzero, so rank saturates at `min(k−1, n)` and stops discriminating. The **effective rank** is a smooth replacement:

$$D_{\text{eff}} = \exp(H(p)), \qquad p_i = \lambda_i / \sum_j \lambda_j,$$

where `λ_i` are the eigenvalues of `C` and `H(p) = −Σ p_i log p_i` is their Shannon entropy. Intuition:

- All variance on one axis → `D_eff = 1`.
- Variance equally spread over `k` axes → `D_eff = k`.
- Uneven spread → something in between, rewarding directions carrying real mass and discounting noise-level directions.

A **continuous idea-count**: how many genuine dimensions of variation the cloud spans.

---

## 3. The three-coordinate profile `P_E(T) = (N, B, I)`

Instead of collapsing the geometry of `X_T` into one number, the paper reports **three coordinates** capturing different facets of "semantic information."

### 3.1 Novelty `N`

$$S_M(T) = \|\mu_T - \mu_0\|_M = \sqrt{(\mu_T - \mu_0)^\top \Sigma_0^{-1} (\mu_T - \mu_0)}$$

$$N(T) = \log(1 + S_M(T)).$$

**In words.** Measure how far the text's centroid sits from the baseline, in Mahalanobis units (baseline-variance-aware), then squash with `log(1 + ·)` so outliers don't dominate.

- `N = 0` ⇒ text indistinguishable from generic prose.
- `N` small ⇒ ordinary prose.
- `N` large ⇒ semantically distant from generic discourse (unusual register, specialized domain).

The answer to: *"How unusual is this text?"*

### 3.2 Breadth `B`

Computed on the **deduplicated** set `T̃`, not the raw cloud `X_T`. This enforces that duplicates can't inflate it.

$$R(\tilde{T}) = \frac{1}{m} \sum_{j=1}^{m} \bigl[1 - \cos(c_j, \mu_{\tilde{T}})\bigr]$$

$$B(T) = D_{\text{eff}}(\tilde{T}) \cdot R(\tilde{T}).$$

Two factors multiplied:
- **`D_eff`** — effective rank of the centroid cloud's covariance. *How many independent directions* the ideas span.
- **`R`** — mean cosine distance of each centroid to the centroid-cloud's mean. *How far apart* the ideas sit within those directions.

Multiplication ensures that genuine breadth requires **both** many directions **and** meaningful spread. The answer to: *"How many genuinely different ideas does this text cover?"*

### 3.3 Integration `I`

$$I_{2\text{-NN}}(T) = \frac{1}{m} \sum_j \text{secondmax}_{l \neq j} \cos(c_j, c_l).$$

For each centroid `c_j`, look at its **second-nearest** neighbor (second-most-similar centroid). Average across all centroids.

Why second-nearest rather than nearest? A single near-duplicate pair that survived deduplication would fake 1-NN coherence: both members would pick each other as nearest neighbor with cosine ~1, dragging the whole average up even when the rest of the text is disconnected. 2-NN demands that every idea has **multiple** close relatives, which is what "coherent" actually means. Grid search over 720 configurations confirms 2-NN is the only choice that distinguishes coherent multi-part passages from bags of unrelated facts.

The answer to: *"How connected are the ideas to each other?"*

### 3.4 Why three coordinates, not one

Consider five genres profiled on `all-mpnet-base-v2`:

| Genre | N | B | I |
|---|---|---|---|
| Poetry (Dickinson) | 3.3 | 0.3 | 0.58 |
| arXiv abstracts | 3.4 | 1.0 | 0.50 |
| EUR-Lex legal | 3.2 | 2.3 | 0.52 |
| Dialogue | 2.9 | 2.7 | 0.24 |
| Code comments | 3.2 | 1.4 | 0.16 |

Poetry: narrow and coherent. Legal: broad and coherent. Dialogue: broad and incoherent. Code comments: disconnected fragments. No single scalar separates these — the three-vector does.

---

## 4. The axiomatic foundation: the SIL theorem

Before introducing the three-coordinate profile, the paper asks: *if you demand a single scalar `I: Text → ℝ` with natural properties, what form must it take?*

### 4.1 The six axioms

Within a fixed frame `(E, μ_0)` — i.e., with the embedding model and baseline held constant — require:

1. **Paraphrase invariance.** If two texts land at (approximately) the same point in embedding space, they get the same `I`. *Outsources the meaning-equality judgment to `E`.*
2. **Redundancy non-increase.** Duplicating a segment never increases `I`. *Meaning ≠ volume.*
3. **Novelty monotonicity.** For fixed covariance, `I` is monotone in `||μ_T − μ_0||`. *Farther from generic ⇒ at least as much information.*
4. **Idea additivity.** For texts in orthogonal embedding subspaces, `I(T₁ ⊕ T₂) = I(T₁) + I(T₂)`. *Semantic analogue of Shannon additivity; unrelated ideas combine additively.*
5. **Orthogonal invariance.** `I` is invariant under rotations of embedding space. *No privileged axes.*
6. **Continuity.** `I` is continuous in the embeddings. *Encoder noise should not cause discontinuous jumps.*

Notation `⊕` means text concatenation, asserted as additive specifically when the two texts live in orthogonal subspaces (so the combined covariance is block-diagonal — a geometric direct sum).

### 4.2 The uniqueness theorem (SIL)

Under axioms 1–6, any such measure has the form, up to a positive scale constant,

$$I_E(T) = \|\mu_T - \mu_0\| \cdot \text{rank}(C_T).$$

**Proof sketch.**
- Axioms 3 and 5 reduce dependence on `μ_T` to the single scalar `S(T) = ||μ_T − μ_0||`.
- Axiom 5 reduces dependence on `C_T` to its spectrum (eigenvalues).
- Axiom 2 eliminates eigenvalue magnitudes (duplication rescales them without adding new directions), leaving `rank(C_T)`.
- Axiom 4 imposes a Cauchy functional equation on the joint dependence; continuity (axiom 6) forces linear solutions. A second Cauchy equation yields the product form.

### 4.3 Why SIL isn't the final answer

The theorem is a **frame-conditional uniqueness result**, not a universal law. And empirically, the factor `rank(C_T)` **saturates** — in 768-dim space, every text with `k` segments scores `rank = k−1`, because encoder noise makes almost every eigenvalue strictly nonzero. The factor carries no signal. Meanwhile `||μ_T − μ_0||` varies richly. So the product **dominates on displacement** alone, collapsing a nominally two-coordinate scalar into effectively one coordinate.

That inadequacy is the motivation for replacing SIL with the three-coordinate profile of §3, where `rank` becomes `D_eff` (smooth, informative) and more structural information enters via `R` and `I`.

---

## 5. The semantic quantum

§2.8 introduced deduplication via a threshold `τ`. The paper elevates this to a **discrete-unit interpretation**.

### 5.1 Definition

Given text `T` with segment embeddings `X_T` and threshold `τ`, the **quantum set** is `Q_τ(T) = T̃ = {c₁, …, c_m}` — the centroids from agglomerative clustering at cosine threshold `1 − τ`. Each `c_j` is a **semantic quantum** at resolution `τ`. The **quantum count** is `m_τ(T) = |Q_τ(T)|`.

### 5.2 Three regimes

- **Sub-quantum (`m_τ ≤ 1`):** single centroid. Novelty defined, breadth = 0, integration conventionally maximal. The text is a point, not a structure.
- **Single-quantum (`m_τ = 2`):** two centroids. Minimal configuration for which all three coordinates are non-trivially defined.
- **Multi-quantum (`m_τ ≥ 3`):** generic regime where breadth and integration vary independently.

### 5.3 `τ` as measurement resolution

Parallel to physical measurement: below scale `1 − τ`, the apparatus cannot resolve separate semantic units. `τ` is the Planck-like resolution parameter of the framework. Default:
- `τ = 0.70` for natural prose (literature, journalism).
- `τ = 0.90` for adversarial near-duplicate stress tests.

This is a **representational commitment** — made explicit rather than hidden.

---

## 6. The no-go theorem: trade-off triangle

Once you have the profile `(N, B, I)`, how do you collapse it to a single number? The paper proves: **you can't do it well.** No scalar can satisfy more than one of three natural desiderata.

### 6.1 The scalar family

Consider every scalar of the form

$$S(T; \Phi) = \Phi\bigl(\varphi_N(N(T); \mathcal{R}),\ \varphi_B(B(T); \mathcal{R}),\ \varphi_I(I(T); \mathcal{R})\bigr)$$

where `R` is a reference set of profiles used for per-coordinate normalization. Two normalizations matter:

- **Min-max:** `φ(x) = (x − min R_X) / (max R_X − min R_X)`. Linear, Lipschitz, but compressed by outliers.
- **Rank:** `φ(x) = rank_R(x) / |R|`. Bounded and outlier-proof, but piecewise constant (discontinuous at every rank boundary).

### 6.2 The three desiderata

**(A) Analytic stability.** Paraphrases get close values (Lipschitz continuity), and `S(T₁ ⊕ T₂)` is a continuous function of `S(T₁), S(T₂)` (closed-form composition).

**(O) Ordinal robustness.** On a benchmark of ordinal discrimination tasks, the scalar ranks categories correctly under noise; bootstrap pass rate exceeds `0.5 + δ_O`; Benjamini–Hochberg correction survives.

**(R) Cross-representation comparability.** Under different embeddings `E₁, E₂` with CKA ≥ `c`, the scalars agree at Spearman ρ ≥ `g(c)`, using a **shared reference set** without per-embedding refitting.

### 6.3 The theorem

For any reasonable tolerances, no scalar in the family satisfies more than one of (A), (O), (R) simultaneously.

**Sketch of the three exclusions.**

- **(A) ⟹ ¬(O).** (A) requires Lipschitz `φ`. Among reference-normalized schemes, only min-max is Lipschitz. Min-max compresses interiors whenever `R` contains outliers — which it always does on a non-degenerate benchmark. Compression prevents ordinal discrimination. So (A) ⇒ ¬(O).
- **(O) ⟹ ¬(A).** (O) requires rank-based `φ` (invariant to monotonic bootstrap distortions). Rank is discontinuous ⇒ fails Lipschitz. Rank also has no closed-form composition: `rank(T₁ ⊕ T₂)` depends on where the concatenation lands in `R`, not on `rank(T₁)` and `rank(T₂)` alone. So (O) ⇒ ¬(A).
- **(R) ⟹ ¬(A), ¬(O).** Shared-`R` across embeddings forces non-identity re-scaling under different `E`, breaking both Lipschitz and ordinal agreement on non-extreme items.

### 6.4 Empirical confirmation

Every scalar in the family — SIL, `S_minmax`, `S_rank`, single-coordinate scalars under either normalization — lands at **at most one corner** on the actual benchmark. No empirical violation.

### 6.5 What the theorem licenses

Three architectural commitments follow:

1. The **profile `(N, B, I)` is the primary object.** It is cross-representation stable coordinate-wise and preserves information a scalar must destroy.
2. The framework must expose **multiple scalars**, each for a different use case. Not one, not three — **two**: one for (A), one for (O). Corner (R) is unreachable.
3. The **choice of scalar is left to the user.** The framework refuses to pick for you, because no pick is universally right.

---

## 7. Two recommended scalars

Both share a weighted geometric mean form and differ only in normalization.

### 7.1 The shared shape

$$S(T) = \bigl(\tilde{N}^{\alpha} \cdot \tilde{B}^{\beta} \cdot \tilde{I}^{\gamma}\bigr)^{1/(\alpha + \beta + \gamma)}$$

- **Geometric mean, not arithmetic.** Any factor going to zero drives `S` to zero. Prevents one big coordinate from masking a failed one.
- **Normalized exponent** `1/(α+β+γ)` keeps the scale comparable regardless of weight magnitude.
- **Weights `(α, β, γ) = (0.5, 3.0, 1.0)`** — set by grid search over 720 configurations on the synthetic benchmark. Breadth dominates; novelty saturates quickly and gets the smallest weight.

### 7.2 `S_minmax` — for analytic work

Normalization: `φ(x) = (x − min R) / (max R − min R) ∈ [0, 1]`.

Lives at corner **(A)**: Lipschitz ⇒ bounded paraphrase drift, compositionality for concatenation. Use for:
- Summarization-loss decomposition `ΔS = ΔN + ΔB + ΔI` with interpretable magnitudes.
- Theoretical paraphrase-drift bounds.
- Reasoning about `S(T₁ ⊕ T₂)` from `S(T₁), S(T₂)`.

Price: outlier compression ⇒ fails ordinal discrimination.

### 7.3 `S_rank` — for ranking work

Normalization: `φ(x) = ε + (1 − ε) · pct_rank(x; R) ∈ (ε, 1]` with `ε = 0.05` floor (so the geometric mean never collapses to zero).

Lives at corner **(O)**: rank is invariant to monotonic bootstrap perturbation ⇒ strong ordinal discrimination. Use for:
- Leaderboards, document ranking, ordinal benchmarks.
- Comparing texts of very different lengths (immune to outlier influence on `R`).

Price: discontinuous at rank boundaries ⇒ ~2× the paraphrase drift of `S_minmax`, and no closed-form composition.

### 7.4 The choice is structural, not tunable

You **cannot** combine min-max and rank to get both corners. The no-go theorem says so. Pick the scalar that matches your use case.

---

## 8. Empirical validation — scope and shape

All numbers live in `validation.ipynb`; this section describes what the notebook tests, not specific values.

### 8.1 Stability

- **Triplication drift:** concatenate each benchmark item to itself three times; measure how much the scalar moves. Should be ~0 (axiom 2). Is.
- **Paraphrase drift:** hand-constructed paraphrase pairs; point estimates plus bootstrap CIs. `S_minmax` is stable; `S_rank` less so (rank boundaries).

### 8.2 Ordinal benchmark — 28 checks

28 pairwise discrimination tasks (e.g., "coherent multi-part passage > bag of unrelated facts", "multi-idea > single-idea"). Report:
- Point-estimate pass rate.
- Bootstrap mean pass rate over 300 resamples.
- Benjamini–Hochberg-corrected significance at `α = 0.05`.

`S_rank` dominates; `S_minmax` is substantially behind; single-coordinate scalars trail further.

### 8.3 Baselines comparison

Seven baselines run through the same 28-check protocol: type-token ratio, unigram entropy, mean sentence length, raw Euclidean novelty, mean-embedding norm, BERTScore-F, breadth alone. The framework's `S_rank` outperforms all of them.

### 8.4 Cross-model robustness

Three embedding models (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `all-distilroberta-v1`). Report per-coordinate Spearman ρ across models and scalar-level ρ. **Coordinates are more stable across encoders than any scalar summary** — consistent with the profile-first architecture.

### 8.5 External corpora

- **Project Gutenberg:** 5 novels, 507 chapters. Natural-text validation, long-form.
- **External short-form:** STS-B, SICK (sentence similarity), arXiv abstracts, EUR-Lex legal, poetry, dialogue, code comments. Each genre gets a `(N, B, I)` signature matching intuition.
- **Disclaimer for STS-B/SICK:** single-sentence items collapse the profile to raw cosine similarity. These numbers measure the **encoder**, not the framework.

### 8.6 SummEval (summarization-loss validation)

100 CNN/DailyMail source–summary pairs with 4 human axes (coherence, consistency, fluency, relevance). Report Spearman ρ between humans and `cos_dir`, `ΔN`, `ΔB`, `ΔI`.

**Honest outcome.** `cos_dir` (directional fidelity) is the only measure that correlates meaningfully with human ratings (moderate ρ on relevance and coherence). The per-coordinate deltas, which worked on synthetic faithful/partial/lossy/off-topic labels, **do not transfer** to human judgments of machine summaries. Real summarizers move multiple coordinates simultaneously; human raters don't decompose coordinate-wise. Conclusion: for summarization diagnostics against humans, use `cos_dir` as the primary signal.

---

## 9. Variational characterization of breadth

Recall that `B = D_eff · R` was introduced as a geometric recipe. §9 shows it is actually the solution to a clean optimization problem — promoting it from heuristic to derived quantity.

### 9.1 Determinants, briefly

For an `n × n` matrix `A`, `det(A)` measures the **volume of the parallelepiped spanned by its rows.**
- `det = 0` ⇒ rows are linearly dependent ⇒ parallelepiped is flat.
- `|det|` large ⇒ rows point in very different directions ⇒ large volume.
- `|det|` small but nonzero ⇒ rows nearly parallel ⇒ thin sliver.

### 9.2 Determinantal Point Processes (DPPs)

A DPP is a distribution over subsets `S ⊆ {1, …, n}` where `P(S) ∝ det(L_S)`. The restricted matrix `L_S` has large determinant exactly when the selected items are **both high-quality and mutually diverse** — duplicates collapse rows, low-quality items shorten them.

**DPP-MAP** (Maximum A Posteriori): pick the subset with highest probability, i.e., maximize `log det L_S`.

### 9.3 The kernel `L` for this paper

$$L_{ij} = q_i \, q_j \, \max(0, \langle x_i, x_j \rangle)$$

- **Quality** `q_i = exp(||x_i − μ_0||_M / σ)` — per-segment novelty weight (Mahalanobis distance from baseline, z-scored across segments).
- **Similarity** — cosine, clipped at zero for PSD-ness.

Maximizing `log det L_S` simultaneously rewards novel segments (via `q`) and discourages redundant pairs (via cosine similarity).

### 9.4 The objective and its solution

$$S^*(T) = \arg\max_{\varnothing \neq S \subseteq [n]} \log \det L_S.$$

- **`log det` is submodular.** Greedy selection (add the segment with highest marginal gain, stop when gain turns non-positive) gives the standard `(1 − 1/e)` approximation guarantee. **On this paper's benchmark, greedy is in fact ILP-optimal** — matching exhaustive enumeration up to `|S| ≤ 16`.
- The **stopping rule selects `|S|` automatically.** No need to specify "pick 3 sentences" externally.

### 9.5 The decomposition theorem

$$\log \det L_S = 2 \sum_{i \in S} \log q_i + \log \det G_S,$$

where `G_S = [⟨x_i, x_j⟩]` is the cosine Gram matrix of `S`. This is an algebraic identity (pull quality diagonals out of the determinant). It splits the objective into a **quality term** (sum of log-qualities, essentially novelty) and a **pure diversity term** (`log det G_S`).

### 9.6 The correspondence with breadth

Under a near-isotropic regime (pairwise cosines ≈ some constant `c` within tolerance `ε`):

- The spectrum of `G_S` concentrates: one eigenvalue near `1 + (|S|−1)c`, the rest near `1 − c`.
- `D_eff(G_S) ≈ |S|` up to `O(ε) + O(|S| c²)` corrections.
- `R(S)` is monotone increasing in `|S|` and decreasing in `c`.
- Both `log det G_S` and `B_S = D_eff · R` are **strictly increasing in `|S|`** and **strictly decreasing in `c`** — they co-vary.

Defining the parameter-free quantity

$$V(T) := \log \det L_{S^*(T)} - 2 \sum_{i \in S^*(T)} \log q_i$$

(the pure diversity contribution after stripping off quality), the corollary is: **`V` recovers `B` up to a monotonic transformation and `O(ε)` error.** Empirical confirmation: Spearman `ρ(V, B) ≈ 0.985` across 507 Gutenberg chapters (number in notebook) — near-perfect monotonic agreement on natural text.

### 9.7 Free consequences

- **Order invariance (0% drift).** DPP objective is over unordered sets; shuffling sentences doesn't move `V`.
- **Triplication invariance (0% drift).** Duplicated segments produce linearly dependent rows in `L`; determinant drops to zero contribution. Axiom 2 is enforced *structurally*, no deduplication step needed.
- **Parameter-free.** No weights `(α, β, γ)`, no trade-off `λ`. The geometry of volume does the balancing.

### 9.8 Why this matters

Before §9, `B = D_eff · R` was an engineering choice. A skeptical reader could ask: *why that combination?* Answer (before): "it works on the benchmark." Answer (after): **it is (up to near-isotropic error) the log-volume of a diversity-maximizing DPP-MAP subset — the answer to a natural optimization question with nothing to do with the benchmark.**

The formula was not chosen — it was discovered.

---

## 10. Downstream: extractive summarization

The variational form is not just theoretically satisfying — it also works as a practical summarizer.

### 10.1 Setup

For each article, select sentences by greedy DPP log-det maximization; evaluate via ROUGE against the human reference. Baselines: Lead-3 (first three sentences), MMR (classical diversity selector with tuned `λ`), centroid selection, uniform random. Tested on XSum (186 articles).

### 10.2 MMR — the classical baseline

**MMR (Maximal Marginal Relevance).** Greedy selection under:

$$\text{MMR}(x) = \lambda \cdot \text{sim}(x, q) - (1 - \lambda) \cdot \max_{x' \in S} \text{sim}(x, x')$$

— a tunable trade-off between relevance and redundancy. Standard baseline for diverse extractive selection. Uses `λ = 0.5` in the experiments.

### 10.3 Result

- **At DPP's own (variationally-selected) size**, DPP significantly beats MMR on ROUGE-L, without any tunable hyperparameter. The parameter-free size selection is itself part of DPP's contribution.
- **Forced to `k = 3`**, DPP and MMR are statistically tied. Lead-3 beats both — a known artifact of news corpora (lead bias), not a framework failure.

**Interpretation.** DPP's empirical advantage over MMR appears precisely at the subset size the DPP objective itself prefers. Fixing the size externally removes that advantage. This is consistent with the theorem in §9.6: the log-det objective jointly balances `|S|` and pairwise similarity, and fixing one of them breaks the balance.

---

## 11. Architecture summary

Three layers of the theory, in decreasing theoretical status:

| Layer | Role | Status |
|---|---|---|
| **SIL axioms + theorem** (§4) | Six axioms → unique form `||μ_T − μ_0|| · rank(C_T)` within a frame. | Uniqueness within a frame; empirically inadequate (rank saturates); scope-setting, not deliverable. |
| **Profile `(N, B, I)`** (§3) | Three orthogonal geometric coordinates. | Primary output of the framework. Cross-representation stable. Passes ordinal tests. |
| **Two scalars `S_minmax`, `S_rank`** (§7) | Use-case-specific summaries. | Practical conveniences; each picks a single corner of the no-go triangle. |

Why this architecture and not a different one:

1. **One universal scalar is impossible** (no-go theorem, §6).
2. **One universal profile is possible** and captures what a scalar must destroy.
3. **Multiple scalars, chosen per use case,** is the correct response to an impossibility theorem: give the user exactly as many tools as the theorem permits useful ones.

---

## 12. Scope, caveats, and honest limitations

### 12.1 Representation-dependence is a feature

Every quantity in the framework depends on `(E, μ_0, τ, segmentation, reference set)`. This is **not** a defect to be engineered away. Three defenses:

1. **Inspectable commitments.** A representation-free measure would still make these choices — just hidden ones. Making `E` explicit turns hidden assumptions into auditable ones.
2. **Calibratable per domain.** Legal `E` ≠ biomedical `E` ≠ general-prose `E`. Exposing the choice lets practitioners pick what fits.
3. **Theorem-mandated.** The no-go theorem proves corner (R) is unreachable. Representation-free comparability isn't a missing feature — it's provably unavailable.

### 12.2 The SummEval result is mixed

Per-coordinate deltas `(ΔN, ΔB, ΔI)` correlate near-zero with human ratings on machine-generated summaries. The framework's attrition-signature story, as originally envisioned, **does not validate** against human judgment. What survives: `cos_dir` is a competitive unsupervised summarization-quality signal.

### 12.3 The 28-check benchmark is synthetic

The ordinal benchmark is hand-constructed. Definitive validation requires human ratings of semantic richness/coherence/informativeness, not yet done at scale. Some benchmark items (e.g., `single_backprop`) sit awkwardly in embedding space and cause specific failures that are encoder artifacts, not framework failures — but this distinction only becomes clear on close inspection.

### 12.4 Quantum lifetime is non-discriminative on short items

The time-resolved extension (persistence of centroids across sliding windows) is included in the framework but is not discriminative on 5-sentence benchmark items. It is only useful on long-form texts like book chapters.

---

## 13. Glossary (one-line references)

- **`E`** — sentence-embedding model.
- **`T, T_i, e_i`** — text, its segments, their embeddings.
- **`n, k, m`** — embedding dimension, raw segment count, deduplicated segment count.
- **`X_T, μ_T, C_T`** — raw embedding cloud, its mean, its covariance.
- **`T̃, c_j`** — deduplicated centroid set, an individual centroid.
- **`μ_0, Σ_0`** — baseline mean, baseline covariance.
- **`||·||_M`** — Mahalanobis norm.
- **`τ`** — deduplication / measurement-resolution threshold.
- **`N, B, I`** — novelty, breadth, integration coordinates.
- **`S_M`** — Mahalanobis displacement from baseline (pre-log novelty).
- **`D_eff`** — effective rank (`exp` of spectral entropy).
- **`R`** — mean radial cosine distance from centroid mean.
- **`I_{2-NN}`** — mean second-nearest-neighbor cosine similarity among centroids.
- **`S_minmax, S_rank`** — the two recommended scalars.
- **`(α, β, γ)`** — profile weights, fixed at `(0.5, 3.0, 1.0)`.
- **`ε`** — rank-normalization floor, fixed at `0.05`.
- **`cos_dir`** — `cos(μ_T, μ_{T'})`; directional fidelity of a summary.
- **`ΔN, ΔB, ΔI, ΔS`** — coordinate and scalar attrition of a summary.
- **`Q_τ(T), m_τ`** — quantum set at resolution `τ`, quantum count.
- **`L, G_S, q_i`** — DPP kernel, Gram submatrix, per-segment quality.
- **`V(T)`** — DPP log-volume (pure diversity contribution), recovers `B`.
- **MMR** — Maximal Marginal Relevance; classical tunable diversity baseline.
- **DPP-MAP** — most-probable subset under a Determinantal Point Process.

---

## 14. Further reading (paper references)

- `geometric_profile_semantic_information.tex` / `.pdf` — the paper itself; contains the formal statement of all laws, measures, and theorems.
- `validation.ipynb` — every empirical number, reproducible.
