# Advanced Explanation Quality Metrics for Molecular Models

Machine learning models for molecular property prediction (e.g., GNNs, fingerprints, or SMILES-based transformers) require explainability to gain trust. Baseline explanations can be evaluated by coverage (fraction of atoms with nonzero importance) and normalized magnitude (mean(|w|/max|w|)), but these are coarse. We propose a richer per-instance metric comprising multiple sub-scores (all in [0, 1]) to reward concise, coherent, stable, and directionally faithful attributions. Each sub-metric is defined formally below, with formulas and computation steps. We also give model-specific guidance for computing attributions (atom importances) and perturbations in Circular Fingerprint, GraphConv, and ChemBERTa models.

---

## 1) Sparsity and Conciseness

Definition: Sparse explanations highlight as few atoms as possible. Equivalently, the set H of atoms with nonzero attribution should be small. We define a sparsity score S that penalizes broad explanations. Let N be the total number of atoms and K = |H| the number of highlighted atoms.

S = 1 - K/N

so that if all atoms are highlighted (K = N), S = 0 (worst), while if only one atom is highlighted, S is near 1 (best). More generally one can weight by magnitude: e.g., use K_ε = sum_i 1(|w_i| ≥ ε) with threshold ε to ignore tiny attributions. The baseline coverage metric is K/N; our sparsity score is 1 − coverage.

Normalized Magnitude: To complement sparsity, we also consider normalized importance values. Let w_i be the importance weight for atom i.

M = (1/K) * sum_{i∈H} |w_i| / max_j |w_j|

A high M means each highlighted atom has weight close to the maximum. (If the baseline use of M is already implemented, it can be included as a sub-metric.)

Interpretation: S close to 1 means very concise (few atoms). We may combine S and M (e.g., by averaging) to capture both fraction and relative strength of attributions.

Computation (pseudocode):

```python
atoms = list_of_atoms(molecule)
w = attribution_weights  # e.g., from model explainer
K = sum(1 for i in atoms if abs(w[i]) > 0)
N = len(atoms)
sparsity = 1 - K / N
max_w = max(abs(w[i]) for i in atoms) if N > 0 else 1.0
norm_mag = (1 / K) * sum(abs(w[i]) / max_w for i in atoms if abs(w[i]) > 0) if K > 0 else 0.0
```

Note: No citation is needed to define sparsity; the idea that sparse explanations (small K) improve interpretability is widely adopted. L0-norm (count of features) is a common proxy for explainability.

---

## 2) Fragment/Substructure Coherence

Definition: Chemists find contiguous substructures more meaningful than scattered atoms. We measure how well the highlighted atoms form connected fragments in the molecular graph. Let G be the molecular graph and H the set of highlighted atoms. Compute the connected components of the subgraph induced by H. Let c be the number of connected components and let L = max_{component C⊆H} |C| be the size of the largest connected highlighted subgraph. We define a coherence score as:

C = L / K

where K = |H| as above. C = 1 if all highlighted atoms lie in a single connected fragment; C < 1 if there are multiple fragments (smaller is worse). One could also define C = (K − c) / (K − 1); for simplicity, we use L/K.

Rationale: If importance is distributed over multiple disconnected pieces, the explanation is less cohesive. The largest-fragment fraction L/K rewards single contiguous clusters. For example, if K = 5 atoms split into one fragment of 3 and two of 1, then L/K = 3/5 = 0.6. If all 5 are connected, C = 1.

Computation (pseudocode):

```python
import networkx as nx
G = molecule_graph(molecule)  # RDKit Mol graph → NetworkX graph
H = [i for i in G.nodes if w[i] != 0]
subG = G.subgraph(H)
components = list(nx.connected_components(subG))
L = max((len(comp) for comp in components), default=0)
K = len(H)
coherence = (L / K) if K > 0 else 0.0
```

We assign C = 0 if H is empty.

---

## 3) Stability

Definition: A good explanation is robust to small perturbations of the input. Here, “perturbation” means a minor change to the molecule representation that should not meaningfully alter the predicted property. In practice, we generate M perturbed versions of the same molecule (e.g., SMILES randomization, feature jitter, or benign atom masking) and compute explanations for each. Let E0 be the original explanation mask (vector of atom importances) and Ej the mask for perturbation j. We measure stability by similarity between Ej and E0.

Define a distance D(E0, Ej) between explanation masks (e.g., cosine distance or L1-difference normalized by |E0|_1). Let instability be max_j D(E0, Ej). Convert to stability St in [0, 1] by:

St = 1 − max_j D(E0, Ej)

(Use a distance that is bounded in [0, 1]; otherwise normalize.) Alternatively, use the average distance.

Rationale: If minor changes drastically change the explanation, trust is undermined. We thus reward high similarity (low D). Perfect stability yields St = 1; complete flip yields St = 0.

Computation (pseudocode):

```python
from sklearn.metrics.pairwise import cosine_distances

def cosine_distance(a, b):
    # both as 2D row vectors
    return float(cosine_distances(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def compute_stability_score(molecule, explain_func, perturb_fn, M=10):
    E0 = explain_func(molecule)
    D_max = 0.0
    for _ in range(M):
        mol_j = perturb_fn(molecule)
        Ej = explain_func(mol_j)
        D = cosine_distance(E0, Ej)  # assume in [0, 1]
        D_max = max(D_max, D)
    return 1.0 - D_max
```

Model-specific perturbations: GraphConv – add small Gaussian noise to node features or drop a non-critical edge; ChemBERTa – alternative SMILES enumerations; Fingerprint – flip non-informative bits. Ensure label/output change is small while probing explanation stability.

---

## 4) Directional Faithfulness

Definition: Directional faithfulness checks that highlighted atoms truly drive the prediction in the expected direction. For binary classification, let P0 be the model’s predicted probability for the positive class on the full molecule. Partition highlighted atoms into H⁺ (w_i > 0) and H⁻ (w_i < 0). Remove H⁺ and recompute probability P−. Remove H⁻ and recompute P+.

Expectation:
- Removing H⁺ should decrease P (P− < P0).
- Removing H⁻ should increase P (P+ > P0).

Define normalized changes (clipped at 0):

Δ⁺ = max(0, (P0 − P−) / max(P0, ε))

Δ⁻ = max(0, (P+ − P0) / max(1 − P0, ε))

Then define faithfulness:

F = 0.5 × (Δ⁺ + Δ⁻)

Thus F ∈ [0, 1]. High values indicate the explanation has causal impact with the correct polarity.

Computation (pseudocode):

```python
P0 = model_predict_proba(molecule)
H_plus = [i for i in H if w[i] > 0]
H_minus = [i for i in H if w[i] < 0]
mol_no_plus = remove_atoms(molecule, H_plus)
mol_no_minus = remove_atoms(molecule, H_minus)
P_minus = model_predict_proba(mol_no_plus)
P_plus = model_predict_proba(mol_no_minus)
EPS = 1e-8
delta_pos = max(0.0, (P0 - P_minus) / max(P0, EPS))
delta_neg = max(0.0, (P_plus - P0) / max(1.0 - P0, EPS))
faithfulness = 0.5 * (delta_pos + delta_neg)
```

Note: “Removal” depends on the model (mask nodes/edges for GNNs; zero bits for fingerprints; delete/mask tokens for SMILES). Keep chemistry valid where possible.

---

## 5) Composite Score

Combine sub-scores into a single Explanation Quality Score Q ∈ [0, 1]. With weights w summing to 1:

Q = wS·S + wC·C + wSt·St + wF·F

Equal weights (all 0.25) are a reasonable default. Report each sub-score alongside Q so users can see which criterion limits quality.

---

## 6) Model-Specific Contribution and Perturbation Strategies

- Circular (Morgan) Fingerprint Model:
  - Attribution: permutation importance of active bits. Map bits → atoms via RDKit bitInfo; distribute bit contributions to atoms.
  - Perturbations: random bit flips/shuffles; or remove mapped atoms and recompute.

- GraphConv (GNN) Model:
  - Attribution: per-atom fragment probability minus whole-molecule probability (w_i = P(frag_i) − P(whole)) or node masking.
  - Perturbations: remove/mask random non-critical atoms/edges; small feature noise.

- ChemBERTa (Transformer on SMILES):
  - Attribution: [CLS] → token attention (or IG on embeddings); map tokens → atoms; aggregate per atom.
  - Perturbations: alternative SMILES enumerations; mask uninformative tokens.

Implementation tips:
- Use the same trained model for perturbation evaluations (no retraining).
- Normalize attribution vectors before distance comparisons.
- When removing atoms, sanitize molecules and re-generate inputs consistently across model types.

---

### Practical Use

- Compute S, M, C, St, F per molecule and average across a test set.
- Tune weights (wS, wC, wSt, wF) per application needs.
- Present a dashboard with sub-scores and Q to diagnose explanation quality across models.
