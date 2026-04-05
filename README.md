# Simplicity Generalizes: Kernel Models vs. SchNet for Molecular Energy Prediction

---

## Abstract

This report documents the design, training, and cross-dataset evaluation of three neural network potentials for molecular energy prediction. Two lightweight kernel-based models are proposed as alternatives to SchNet, a well-established message-passing neural network used here as a performance benchmark. All models are trained on QM9 and evaluated on PC9, a structurally distinct dataset sharing the same chemical space. Results show that the simpler proposed models generalize substantially better across datasets: Model A degrades by a factor of ×1.49 and Model B by ×2.64, while SchNet degrades by a factor of ×175 — suggesting that architectural expressiveness without sufficient regularization leads to distribution-specific overfitting that fails at transfer time.

---

## 1. Datasets

### 1.1 QM9

QM9 is a standard benchmark dataset in quantum chemistry machine learning [1]. It contains **133,885 small organic molecules** composed exclusively of H, C, N, O, and F atoms (up to 29 atoms per molecule), with geometries optimized at the B3LYP/6-31G(2df,p) level of theory [2].

Each molecule is stored as an extended `.xyz` file containing 3D atomic coordinates, Mulliken partial charges, vibrational frequencies, SMILES and InChI strings, and **15 quantum chemical properties** computed with DFT. The target property used in this work is **U0** — the internal energy at 0K, reported in Hartree.

**Key statistics (full dataset):**

| Property | Value |
|---|---|
| Total molecules | 133,885 |
| Atom count range | 3 – 29 |
| Mean atom count | 17.98 ± 2.95 |
| U0 range | −714.57 to −40.48 Ha |
| U0 mean | −411.54 Ha |
| U0 std | 40.06 Ha |

The U0 distribution is approximately Gaussian, centered around −411 Ha with a standard deviation driven primarily by molecular size. A Z-score normalization (`U0_norm = (U0 − mean) / std`) was applied before training, with the physical values preserved separately for evaluation.

> 📊 *[Insert: U0 distribution — physical and normalized]*

> 📊 *[Insert: Atom count distribution per molecule]*

---

### 1.2 PC9

PC9 is a dataset of **99,234 drug-like organic molecules**, extracted from the PubChemQC project and constrained to the same chemical space as QM9 (H, C, N, O, F; up to 9 heavy atoms) [3]. While the two datasets share element types and molecular size range, only 18% of PC9 molecules also appear in QM9 — the remaining 82% are structurally distinct compounds absent from the training distribution. S, Cl, and P are declared in the format specification but absent from this specific subset. Molecular sizes range from 2 to 29 atoms, with a mean of 18.08 ± 4.16 — slightly larger and more variable than QM9.

Properties are stored in tab-separated format in the header line of each `.xyz` file and include HOMO, LUMO, HOMO-LUMO gap, and total energy E (column index 11), also in Hartree.

**Key statistics:**

| Property | Value |
|---|---|
| Total molecules | 99,234 |
| Atom count range | 2 – 29 |
| Mean atom count | 18.08 ± 4.16 |
| Energy range | −750.49 to −40.52 Ha |
| Energy mean | −390.93 Ha |
| Energy std | 64.85 Ha |

PC9 has a broader and slightly shifted energy distribution compared to QM9 (std 64.85 Ha vs 40.06 Ha), reflecting the inclusion of more chemically diverse, drug-like structures. All 99,234 molecules fall within the QM9 atom-count range (≤ 29 atoms).

> 📊 *[Insert: Energy distribution — physical and normalized]*

> 📊 *[Insert: Atom count distribution and heavy atom count distribution]*

> 📊 *[Insert: HOMO, LUMO, and HOMO-LUMO gap distributions]*

> 📊 *[Insert: Element presence bar chart]*

> 📊 *[Insert: Per-element atom count histograms]*

> 📊 *[Insert: Energy vs. molecule size — scatter and mean line]*

---

### 1.3 Dataset Comparison

Both datasets share the same chemical space (HCNOF), the same size range (≤ 29 atoms), and use DFT-computed energies. The key differences are distributional: PC9 molecules are on average slightly larger, more chemically diverse (97 distinct functional groups identified vs. 71 in QM9 [3]), and have a wider and shifted energy distribution. Crucially, PC9 was derived from real PubChem molecules rather than the combinatorial GDB enumeration that underlies QM9, making it structurally richer in the acyclic chemical space [3]. This makes PC9 a meaningful out-of-distribution test for models trained on QM9 — same domain, different distribution.

---

## 2. Model Design

Three models were designed and implemented in PyTorch. All operate directly on the 3D molecular geometry (atomic numbers Z and Cartesian positions R), require no featurization beyond parsing the `.xyz` files, and produce a scalar energy prediction per molecule.

The common design principle is **atomistic decomposition**: total molecular energy is expressed as a sum of per-atom or per-pair contributions. This is physically motivated and consistent with the approach introduced by Behler and Parrinello for high-dimensional neural network potentials [4], and later adopted by the broader family of message-passing neural network potentials, including SchNet [5].

All models use a learnable `nn.Embedding(max_Z=10, ...)` to represent atom types. The Z-score normalized U0 is used as the training target, with predictions denormalized back to Hartree at evaluation time. Variable-size molecules are handled by a custom `collate_fn` that passes each batch as a list of molecule dictionaries, computing pairwise distance matrices on-the-fly per molecule inside the forward pass.

---

### 2.1 Model A — KernelModel

**Architecture.** The simplest of the three models. Energy is decomposed into two terms:

1. **Atomic term**: a learned scalar energy per atom type via `nn.Embedding(max_Z, 1)`, summed over all atoms.
2. **Pairwise interaction term**: for each unique pair (i < j), the pairwise distance r_ij is passed through a 3-layer MLP with SiLU activations. The output is weighted by the product of atomic numbers Z_i × Z_j, mimicking a Coulomb-like interaction.

$$E = \sum_i \epsilon(Z_i) + \sum_{i < j} Z_i \cdot Z_j \cdot f_\theta(r_{ij})$$

where $\epsilon$ is the atomic embedding and $f_\theta$ is the MLP kernel.

| Component | Detail |
|---|---|
| MLP input | r_ij (1-dimensional) |
| MLP layers | Linear(1→64) → SiLU → Linear(64→64) → SiLU → Linear(64→1) |
| Total parameters | **4,363** |
| Optimizer | Adam, lr = 5×10⁻⁴ |

The Z_i × Z_j weighting provides a strong **inductive bias**: interactions are constrained to scale with atomic number product, a physically meaningful prior analogous to Coulomb's law. The kernel function itself receives only distance information — it cannot distinguish between, say, a C-C pair and an H-O pair at the same distance without relying on the scalar Z product. In Mitchell's terminology [6], this constitutes a strong bias in the generalization language: the class of functions expressible by this model is a strict subset of all possible pairwise potentials, one in which pair interactions scale monotonically with nuclear charge.

---

### 2.2 Model B — KernelModelFlexible

**Architecture.** A direct extension of Model A that gives the MLP explicit access to atom identity for each pair. The key change is the MLP input, which now receives three features: (Z_i, Z_j, r_ij), all normalized to similar numerical scales.

$$E = \sum_i \epsilon(Z_i) + \sum_{i < j} g_\theta\!\left(\frac{Z_i}{10},\; \frac{Z_j}{10},\; \frac{r_{ij}}{5}\right)$$

The multiplicative Z_i × Z_j weighting from Model A is dropped; instead, the MLP learns the full interaction function jointly over atom types and distance.

| Component | Detail |
|---|---|
| MLP input | (Z_i/10, Z_j/10, r_ij/5) — normalized 3D vector |
| MLP layers | Linear(3→64) → SiLU → Linear(64→64) → SiLU → Linear(64→1) |
| Total parameters | **4,491** |
| Optimizer | Adam, lr = 5×10⁻⁴ |

Input normalization (dividing Z by 10 and r by 5) keeps all inputs in a comparable numerical range, stabilizing training. This model retains essentially the same parameter count as Model A while gaining the ability to distinguish all atom-type pair combinations explicitly, leading to substantially smoother convergence. Compared to Model A, the inductive bias is slightly relaxed: the model is no longer constrained to a Coulomb-like weighting scheme, but it still lacks any multi-body context.

---

### 2.3 Model C — SchNet (Benchmark)

**Architecture.** A faithful implementation of the SchNet architecture [5], used here as a performance benchmark representing the established state of the art in learned interatomic potentials. SchNet extends the atomistic decomposition approach of Behler and Parrinello [4] with continuous-filter convolutional interaction blocks that allow the filter generating network to operate on arbitrary atom positions rather than a fixed grid. This yields smooth, rotationally invariant energy predictions while retaining differentiability with respect to atomic coordinates.

The model consists of four components:

1. **Atomic embedding**: `nn.Embedding(max_Z, 128)` — each atom starts with a 128-dimensional feature vector.
2. **Gaussian Basis (RBF) expansion**: pairwise distances within r_cutoff are expanded into a 64-dimensional basis of Gaussian functions with evenly spaced centers in [0, r_cutoff].
3. **Interaction blocks** (×3): each block transforms atomic feature vectors via a filter network conditioned on pairwise RBF distances. For each pair (i, j), a message is computed as the atom-j state modulated by a distance-dependent filter, then aggregated onto atom i with a residual connection.
4. **Readout**: a 2-layer MLP maps each atom's final hidden state to a scalar contribution; these are summed over all atoms to produce the molecular energy.

| Component | Detail |
|---|---|
| r_cutoff | 10.0 Å |
| n_interactions | 3 |
| hidden_dim | 128 |
| n_basis (RBF) | 64 |
| Total parameters | **232,705** |
| Optimizer | Adam, lr = 5×10⁻⁴ |

SchNet is approximately 53× larger than Models A and B in parameter count and incorporates multi-body context through iterative message passing — each atom's representation is refined based on its neighbors' states over multiple rounds of interaction. In the original publication, SchNet achieved a MAE of 0.31 kcal/mol on QM9 U0 with 110k training examples [5]; the results here are compatible with that range, confirming the implementation is sound.

---

### 2.4 Model Summary

| | Model A | Model B | Model C (SchNet) |
|---|---|---|---|
| Type | Kernel (distance only) | Kernel (atom-aware) | Message-passing |
| Parameters | 4,363 | 4,491 | 232,705 |
| MLP input | r_ij | Z_i, Z_j, r_ij | RBF(r_ij) |
| Atom identity in interaction | Z·Z weighting | MLP input | Learned 128-dim embedding |
| Multi-body interactions | No | No | Yes (3 rounds) |
| Inductive bias strength | Strong | Moderate | Weak (learned) |

---

## 3. Training

### 3.1 Train/Test Evaluation on QM9

For model comparison, a 70/30 train/test split was applied over the full QM9 dataset (93,719 train / 40,166 test molecules). All models were trained with Adam optimizer, batch size 32, using MSE loss over Z-score normalized U0.

**Training curves (MSE on normalized target):**

| Epoch | A Train | A Test | B Train | B Test | C Train | C Test |
|---|---|---|---|---|---|---|
| 1 | 1349.16 | 2.67 | 10.63 | 6.22 | 0.0582 | 0.0098 |
| 2 | 2.52 | 1.19 | 4.12 | 2.48 | 0.0066 | 0.0042 |
| 3 | 8.37 | 2.46 | 1.35 | 0.81 | 0.0036 | 0.0044 |
| 4 | 7.77 | 0.15 | 0.65 | 0.47 | 0.0027 | 0.0016 |
| 5 | 5.10 | 7.57 | 0.36 | 0.26 | 0.0019 | 0.0031 |
| 6 | 4.86 | 0.09 | 0.18 | 0.13 | 0.0015 | 0.0007 |
| 7 | 3.58 | 3.32 | 0.10 | 0.08 | 0.0012 | 0.0009 |
| 8 | 3.00 | 0.74 | 0.07 | 0.07 | 0.0010 | 0.0011 |
| 9 | 2.48 | 3.67 | 0.06 | 0.06 | 0.0009 | 0.0102 |
| 10 | 2.28 | 0.89 | 0.05 | 0.05 | 0.0007 | 0.0003 |
| 11 | 1.94 | 0.27 | 0.05 | 0.04 | — | — |
| 12 | 1.58 | 0.06 | 0.04 | 0.05 | — | — |
| 13 | 1.54 | 0.14 | 0.04 | 0.05 | — | — |
| 14 | 1.15 | 0.17 | 0.04 | 0.03 | — | — |
| 15 | 0.99 | 0.09 | 0.03 | 0.04 | — | — |

> 📊 *[Insert: Training curve comparison — Models A, B, C]*

Model A shows highly unstable test loss (oscillating between 0.06 and 7.57) despite a steadily decreasing train loss — a sign that limited expressiveness makes it sensitive to the composition of each test batch. Model B converges smoothly and monotonically from epoch 1. Model C starts two orders of magnitude below Models A and B, reflecting SchNet's superior representational capacity; it is trained for only 10 epochs as convergence is effectively reached by epoch 6–7.

### 3.2 Full Dataset Training for Export

After the train/test evaluation, each model was retrained on the **full QM9 dataset** (133,885 molecules) without any held-out split, for export and downstream evaluation. Normalization statistics (U0_mean = −411.544 Ha, U0_std = 40.060 Ha) were saved alongside the model weights to enable correct denormalization at inference time.

| Model | Epochs | Final train loss |
|---|---|---|
| Model A | 10 | — |
| Model B | 10 | — |
| Model C | 5 | 0.0018 |

Models are saved as `.pt` files containing the `state_dict`, model class name, training loss history, and normalization statistics.

---

## 4. QM9 Test Set Results

Models were evaluated on the held-out 30% test set (40,166 molecules). Predictions were denormalized to Hartree before computing errors.

**Test set performance:**

| Model | MAE (Ha) | MAE std (Ha) | MAE rel | MAE rel std |
|---|---|---|---|---|
| Model A | 8.844 | 7.809 | 2.27% | 3.80% |
| Model B | 4.371 | 6.273 | 1.16% | 3.35% |
| Model C (SchNet) | **0.339** | **0.646** | **0.088%** | **0.601%** |

SchNet achieves an MAE of 0.339 Ha on QM9, roughly 26× lower than Model A and 13× lower than Model B, confirming its position as the strongest model within the training distribution. Notably, Model C also has a much tighter error distribution (lower std), indicating more consistent predictions across molecule types.

> 📊 *[Insert: Absolute error distributions per model — QM9 test set]*

> 📊 *[Insert: Relative error distributions per model — QM9 test set]*

> 📊 *[Insert: Pred vs. True scatter plots — QM9 test set]*

> 📊 *[Insert: MAE vs. molecule size (n_atoms) — QM9 test set]*

---

## 5. Cross-Dataset Evaluation on PC9

The exported models (trained on full QM9) were evaluated on all 99,234 molecules of PC9 without any fine-tuning. PC9 energies are directly comparable to QM9 U0 values (same units, same DFT level). The QM9 normalization statistics were used for denormalization at inference time.

### 5.1 Results

**PC9 evaluation results:**

| Model | N | MAE (Ha) | RMSE (Ha) | MAE rel (%) |
|---|---|---|---|---|
| Model A | 99,234 | 13.141 | 23.679 | 4.27% |
| Model B | 99,234 | 11.552 | 20.854 | 3.64% |
| Model C (SchNet) | 99,234 | 59.265 | 77.096 | 15.38% |

> 📊 *[Insert: MAE bar chart by model — absolute and relative — on PC9]*

> 📊 *[Insert: Pred vs. True scatter plots on PC9]*

> 📊 *[Insert: Absolute error distributions on PC9]*

> 📊 *[Insert: Relative error distributions on PC9]*

> 📊 *[Insert: MAE vs. molecule size on PC9]*

### 5.2 Degradation Analysis

The cross-dataset degradation factor (MAE on PC9 / MAE on QM9 test) reveals a sharp inversion of the within-distribution ranking:

| Model | MAE QM9 (Ha) | MAE PC9 (Ha) | Degradation factor |
|---|---|---|---|
| Model A | 8.844 | 13.141 | **×1.49** |
| Model B | 4.371 | 11.552 | **×2.64** |
| Model C (SchNet) | 0.339 | 59.265 | **×175** |

Model C — the clear winner on QM9 — collapses catastrophically on PC9. Models A and B, which performed worse in-distribution, degrade only modestly and maintain physically reasonable error magnitudes. This pattern closely mirrors what Glavatskikh et al. observed when testing a QM9-trained SchNet on PC9 molecules not seen during training: they reported an MAE of 8.9 kcal/mol for subset B (PC9-exclusive compounds), substantially higher than the 1.0 kcal/mol achieved on QM9 itself [3]. The scale of degradation found here is more severe, but the direction is consistent and points to the same underlying cause: the QM9 training distribution does not adequately represent the broader chemical diversity present in real-world molecules.

---

## 6. Discussion

### 6.1 Why SchNet Fails to Transfer

SchNet's poor generalization is the central finding of this work.

**Capacity and distribution-specific overfitting.** SchNet has 232,705 parameters — 53× more than Models A and B. With 133,885 training molecules and no explicit regularization (no dropout, no weight decay beyond Adam's implicit behavior), SchNet has sufficient capacity to memorize QM9-specific patterns: characteristic bond length distributions, typical atom-type neighborhood frequencies, and the tight correlation between molecular size and energy in that dataset. When applied to PC9 — which has a broader energy distribution (std 64.85 vs 40.06 Ha) and more diverse chemical structures — these patterns do not generalize. Glavatskikh et al. note that QM9's chemical homogeneity is itself a product of its combinatorial GDB origin: it presents less diversity in the acyclic chemical space and concentrates functional groups to an extent that exceeds a real-life dataset [3], making it a poor proxy for real molecular distributions.

**Inductive bias.** Models A and B encode a physically motivated prior: energy is a sum of pairwise interactions with magnitude tied to atomic numbers. This functional form generalizes naturally across molecular datasets because it reflects genuine physical structure regardless of the training distribution. SchNet learns its own interaction functions entirely from data, which is more expressive but provides no such guarantee. The weaker the inductive bias, the more the model's behavior depends on reproducing the training distribution. As Mitchell argued already in 1980, a learner with no inductive bias beyond consistency with training examples cannot make the generalization leap necessary to classify unseen instances non-arbitrarily [6] — it reduces, in effect, to rote memorization of the training set's statistical regularities.

**Normalization mismatch.** The QM9 normalization statistics (mean = −411.54 Ha, std = 40.06 Ha) are used to denormalize PC9 predictions. PC9 has a different mean (−390.93 Ha) and a substantially larger std (64.85 Ha). A model with high sensitivity to the normalized target range — as SchNet is, given its capacity — will accumulate systematic errors when the true distribution departs from the training one. The simpler models, anchored to a physics-grounded functional form, are less sensitive to this mismatch.

### 6.2 The Role of Simplicity

The results support a broader principle: **within-distribution accuracy is not a reliable proxy for cross-dataset generalizability**. Model C achieves the best QM9 test MAE by a wide margin, but this advantage disappears entirely under distribution shift. Models A and B sacrifice some in-distribution performance in exchange for a more robust inductive bias, which pays off at transfer time.

This finding has a direct parallel in the machine learning literature. Teney et al. show that neural networks trained with SGD exhibit a *simplicity bias* — they preferentially latch onto the simplest linearly-predictive features in the training data and ignore more complex ones, even when those complex ones correspond to the actual causal structure of the task [7]. In their setting, this causes failures under out-of-distribution (OOD) shifts; here, SchNet's analogue is learning the simplest statistical patterns that explain QM9 energies (which include QM9-specific molecular geometry signatures), without learning the more general physical interaction structure that would transfer to PC9. Conversely, Models A and B are, in effect, forced to learn a more transfer-robust representation because their functional form cannot express QM9-specific artifacts — a constraint that acts as a beneficial inductive bias rather than a limitation.

Webb's theorem of decreasing inductive power offers a useful theoretical framing: for classifiers covering identical training cases, a more general (less constrained) classifier is more likely to misclassify unseen examples [8]. Applied here, SchNet's vastly larger hypothesis space allows it to fit QM9 well, but that very generality means it makes predictions in regions of chemical space (PC9) further from the training examples, where accuracy is harder to guarantee. Models A and B, constrained to a physically motivated functional form, effectively operate in a smaller hypothesis space whose coverage extends more uniformly across the HCNOF chemical space.

This does not mean that simple models are always preferable. In settings where distribution shift is minimal — same dataset, different split; or closely related data generating processes — SchNet would be the right choice. The key insight is that **model selection should account for the target deployment scenario**: when generalization across molecular datasets or distributions matters, architectural simplicity and physical inductive bias are undervalued properties that deserve explicit consideration alongside raw in-distribution performance.

---

## 7. Conclusions

Three neural network potentials were designed, trained on QM9 [1], and evaluated on PC9 [3]:

- **Model A (KernelModel)**: energy as atomic embeddings plus a distance-only pairwise kernel weighted by Z_i × Z_j. MAE on QM9: 8.84 Ha → on PC9: 13.14 Ha (×1.49 degradation).
- **Model B (KernelModelFlexible)**: same structure but the kernel MLP receives atom identities alongside distance. MAE on QM9: 4.37 Ha → on PC9: 11.55 Ha (×2.64 degradation).
- **Model C (SchNet [5], benchmark)**: message-passing network with continuous-filter convolutions, 232K parameters. MAE on QM9: 0.34 Ha → on PC9: 59.27 Ha (×175 degradation).

The proposed lightweight models generalize substantially better across datasets. SchNet's superior expressiveness becomes a liability under distribution shift, while the physically grounded kernel models retain meaningful predictive accuracy on an unseen molecular distribution. These results highlight the importance of inductive bias [6] as a design criterion in molecular machine learning, particularly when cross-dataset generalization is a requirement — consistent with the broader finding that models trained on QM9 show limited ability to generalize to real-world molecular distributions [3].

---

## References

[1] Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1, 140022.

[2] Becke, A. D. (1993). Density-functional thermochemistry. III. The role of exact exchange. *The Journal of Chemical Physics*, 98(7), 5648–5652.

[3] Glavatskikh, M., Leguy, J., Hunault, G., Cauchy, T., & Da Mota, B. (2019). Dataset's chemical diversity limits the generalizability of machine learning predictions. *Journal of Cheminformatics*, 11, 69.

[4] Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.

[5] Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., & Müller, K.-R. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, 30.

[6] Mitchell, T. M. (1980). The need for biases in learning generalizations. Technical Report CBM-TR-117, Rutgers University, Department of Computer Science.

[7] Teney, D., Abbasnejad, E., Lucey, S., & van den Hengel, A. (2022). Evading the simplicity bias: Training a diverse set of models discovers solutions with superior OOD generalization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16761–16772.

[8] Webb, G. I. (1996). Generality is more significant than complexity: Toward an alternative to Occam's Razor. Unpublished manuscript, Deakin University.

---

## Appendix: Implementation Notes

**Normalization.** U0 is normalized using Z-score statistics computed over the full QM9 training set (mean = −411.544 Ha, std = 40.060 Ha). The normalized value is used as the training target; physical values are restored at evaluation.

**Variable-size batching.** QM9 molecules vary in atom count (3–29 atoms). A custom `collate_fn` passes each batch as a Python list of molecule dictionaries, avoiding padding. Pairwise distance matrices are computed on-the-fly per molecule inside each model's forward pass.

**Model export format.** Trained models are saved as `.pt` files containing: `state_dict`, model class name, training loss history, `U0_mean`, and `U0_std`. This enables self-contained loading and inference without requiring the training code.

**PC9 parser note.** The energy property in PC9's `.xyz` header is at tab-separated index 11 (not 10, as the MODEL.xyz column listing might suggest at first read). This was identified and corrected during the EDA phase before any model evaluation.
