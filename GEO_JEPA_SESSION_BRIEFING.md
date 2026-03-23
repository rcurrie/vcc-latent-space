# Geo-JEPA Virtual Cell — Coding Agent Session Briefing
## Version 1.0 | Files: build_proxy_dataset.py, geo_jepa_mnist.py

---

## PURPOSE OF THIS DOCUMENT

This briefing gives a coding agent full context to continue development of the
Geo-JEPA Virtual Cell project. Read it entirely before touching any code.
The two attached scripts are a working prototype. Your job is to extend,
debug, or evaluate them as directed in the session task below.

---

## PROJECT OBJECTIVE

Build a foundation model for single-cell RNA sequencing (scRNA-seq) that
outperforms current SOTA (scGPT, scFoundation, STATE) on the Virtual Cell
Challenge (VCC) benchmarks by replacing token-prediction objectives with a
geometrically grounded Joint-Embedding Predictive Architecture (JEPA)
constrained by Maximal Coding Rate Reduction (MCR²).

Target hardware: MacBook Air (Apple Silicon, 16-32GB RAM).
Framework: PyTorch 2.x with MPS backend.
Development phase: Proxy dataset validation before scaling to real scRNA-seq.

---

## THE CENTRAL PROBLEM: THE LINEARIZATION TRAP

Current foundation models (scGPT, scFoundation) treat gene expression like
language: tokenize genes, pretrain with masked token reconstruction, fine-tune
for perturbation prediction. This fails on the VCC for a specific geometric
reason called the Linearization Trap:

    At a bifurcation point (a progenitor cell that can become either
    cell type A or cell type B), a model trained with MSE in latent
    space learns to predict:

        predicted_embedding ≈ E[z_fate]
                            = 0.5 * z_A + 0.5 * z_B

    This interpolant lies BETWEEN the two fate subspaces — it
    corresponds to no real biological state and scores near zero
    on the VCC Perturbation Discovery Score (PDS).

Evidence: Ahlmann-Eltze, Huber & Anders (Nature Methods 2025) showed that
five foundation models and two deep learning models all failed to outperform
a simple linear baseline that predicts the training mean. The training mean
IS the Linearization Trap made explicit.

---

## THE SOLUTION ARCHITECTURE

### Core Principle
Replace the decoder-based reconstruction objective with a JEPA prediction
objective in embedding space. Enforce geometric structure (orthogonal fate
subspaces) via MCR² rather than hoping attention learns it implicitly.

### Two-Stage Encoder

    STAGE 1 — Gene Interaction Prior
        Single multi-head attention layer over expressed genes only.
        Each gene is a token: (gene_identity_embedding + Fourier(expression)).
        Only top-k expressed genes attend (k=200 for MacBook speed).
        Output: h ∈ R^{embed_dim}  — gene-interaction-aware cell embedding.

        Why one attention layer and not a full transformer:
        - Full transformer is O(n²) over 20k genes = intractable
        - We need gene-gene regulatory priors but NOT a full language model
        - The MCR² encoder (Stage 2) handles the geometric structure;
          Stage 1 only needs to surface regulatory co-expression signals

    STAGE 2 — MCR² ReduNet Encoder
        L unrolled gradient steps of the MCR² objective ΔR.
        Each layer is one gradient ascent step:

            Z^(l+1) = Z^(l) + η (E·Z^(l) - Σ_k (n_k/n) C_k·Z_k^(l))

        where:
            E    = (I + ZZ^T / (nε²/d))^{-1}         [expansion: push apart]
            C_k  = (I + Z_k Z_k^T / (n_k ε²/d))^{-1} [compression: per class]
            Z_k  = rows of Z belonging to class k (soft assignment)

        Key property: the encoder IS the MCR² objective unrolled.
        You do NOT need a separate MCR² loss term — it is computed
        inside the forward pass. The MCR² loss in the training loop
        is a diagnostic, not the primary gradient signal.

        Matrix inversions use Cholesky + cholesky_solve (NOT linalg.solve,
        which has MPS bugs). Woodbury identity swaps large inversions for
        small ones when batch_size < embed_dim.

    COMBINED: GeoJEPAEncoder = GeneInteractionPrior → ReduNetEncoder
        Input:  x ∈ R^{gene_dim}  (sparse expression, MNAR dropout applied)
        Output: z ∈ R^{embed_dim} (normalized, lives in union-of-subspaces)
                Pi ∈ R^{n × K}   (soft class assignments, for loss and predictor)

### JEPA Training Objective

    Two encoders:
        encoder_online  — trained via gradient
        encoder_target  — EMA shadow of encoder_online (momentum=0.999)

    Gene-set masking (the scRNA-seq analog of spatial patch masking):
        Partition gene indices into context (75%) and target (25%).
        Context encoder sees context genes; target encoder sees target genes.

    Predictor (FateConditionalPredictor):
        Input:  z_C (context embedding) + Pi_C (soft subspace assignment)
        Output: ẑ_T (predicted target embedding)
        Architecture: 2-layer MLP with GELU + LayerNorm

        CRITICAL DESIGN: Pi_C is fed to the predictor as an explicit gate.
        At a bifurcation point, Pi_C is spread across two fate subspaces.
        The predictor must resolve which subspace to predict into — it cannot
        average across them without landing off-manifold, which is penalized
        by the prediction loss. This is the architectural fix to the
        Linearization Trap at the predictor level.

    Loss:
        L_pred = MSE(ẑ_T, sg(z_T))          [JEPA prediction, main signal]
        L_mcr2 = -ΔR(Z_C, Pi_C)             [MCR² diagnostic, small weight]
        L_total = L_pred + 0.1 * L_mcr2

### EMA Anchor (Phase B Stability — NOT YET IMPLEMENTED)
    When transitioning from Phase A (homeostatic scaffold) to Phase B
    (perturbation learning), an EMA anchor loss prevents catastrophic
    forgetting of the anchor room geometry:

        L_anchor = ||sg(z_base_anchor) - z_base||²

    where z_base_anchor = encoder_target(unperturbed cells) from Phase A.
    This is currently a TODO — Phase A and Phase B are trained jointly
    in the prototype.

---

## THE PROXY DATASET

Real scRNA-seq (VCC benchmark) is not available for initial development.
We use an MNIST-based proxy that approximates the key statistical and
topological properties of the VCC dataset.

### Conceptual Mapping

    MNIST concept                 VCC / scRNA-seq concept
    --------------------------------------------------------
    784-dim pixel vector          ~20k-dim gene expression
    Random projection -> 2000D    High-dimensional sparse gene space
    MNAR dropout (~80% zeros)     Technical dropout (low-expr genes lost)
    NB noise on projected values  Count-based sequencing overdispersion
    7 digit classes               7 cell types (homeostatic "rooms")
    Digit 9 cells                 Pluripotent progenitor (ambiguous fate)
    Blend(digit9, mean_digit4)    CRISPRi knockdown -> erythroid fate
    Blend(digit9, mean_digit1)    CRISPRi knockdown -> lymphoid fate
    Rotation of digit 4           In-fate transcriptional modulation
    30% blend toward fate         Partial fate priming (probabilistic)
    Held-out blend -> digit 7     OOD perturbation (unseen at test time)

### Three Biological Regimes Represented

    Regime 1 (pert_id 0, 1): Fate-determining perturbations
        Progenitor -> committed fate. Discrete room switch.
        Primary test of whether Linearization Trap is avoided.
        Ground truth: pert_id 0 cells should land near digit-4 room.
                      pert_id 1 cells should land near digit-1 room.

    Regime 2 (pert_id 2): In-room modulation
        Digit-4 cells rotated in pixel space. Same cell type,
        different transcriptional program.
        Tests: does the model correctly keep these in the digit-4 subspace?

    Regime 3 (pert_id 3): Partial fate priming
        30% blend toward fate-A. Cell has not committed.
        Tests: does Pi_C spread across two subspaces (correct) or
               does it force a hard assignment (incorrect)?

    Combinatorial (pert_id 4): pert_id 0 + pert_id 2 simultaneously.

    OOD (ood_cells): Blend toward digit 7 — never seen during training.
        Tests: can the model generalize to unseen perturbation paths?
        VCC analog: held-out perturbations in H1 hESC test set.

### MNAR Dropout Model
    P(dropout | x_ij) = sigmoid(-2.0 * expr_normalized_ij + 1.5)
    High-expression genes: ~5% dropout
    Zero-expression genes: ~82% dropout
    Overall sparsity: ~75-85%

    This is DIFFERENT from uniform random masking. The model must learn
    to recover signal from expression-dependent missingness, not just
    random noise.

### Dataset Arrays (proxy_dataset.npz)

    homeostatic_cells       (N_home, 2000)   float32   — Phase A training
    homeostatic_labels      (N_home,)         int32    — class 0-6 (7 rooms)
    homeostatic_digit_ids   (N_home,)         int32    — actual MNIST digit
    perturbation_cells      (N_pert, 2000)   float32   — Phase B training
    perturbation_labels     (N_pert,)         int32    — source room label
    perturbation_ids        (N_pert,)         int32    — type 0-4
    perturbation_strengths  (N_pert,)         float32  — dose 0.3/0.6/1.0
    ood_cells               (N_ood, 2000)    float32   — OOD test
    ood_labels              (N_ood,)          int32    — source room label
    projection_matrix       (784, 2000)      float32   — fixed, for reproducibility
    gene_dim_metadata       scalar            int32    — = 2000

    Load with:
        data = numpy.load("proxy_data/proxy_dataset.npz")
        meta = json.load(open("proxy_data/proxy_dataset_metadata.json"))

---

## KEY EVALUATION METRIC: DISCRIMINATION RATIO

    DR = dist(predicted_embedding, wrong_fate_room) /
         dist(predicted_embedding, correct_fate_room)

    For pert_id 0 (full strength):
        correct room = digit-4 centroid
        wrong room   = digit-1 centroid
        Target: DR > 2.0

    DR < 1.0 means the model predicts CLOSER to the wrong fate than the
    correct one — the Linearization Trap is active.
    DR = 1.0 means the model is equidistant — no discrimination.
    DR > 2.0 means the model correctly commits the progenitor to fate-A.

    This metric is computed at the end of each training run in
    geo_jepa_mnist.py (evaluate_discrimination function).

---

## FILE DESCRIPTIONS

### build_proxy_dataset.py
    Standalone dataset generator. Run first.

    Inputs:  MNIST (downloaded automatically via sklearn)
    Outputs: proxy_data/proxy_dataset.npz
             proxy_data/proxy_dataset_metadata.json

    Key functions:
        load_mnist()                   — downloads MNIST
        make_projection_matrix()       — sparse 784->2000 projection
        project_to_gene_space()        — ReLU + scale -> count-like values
        mnar_dropout()                 — expression-dependent dropout
        add_negative_binomial_noise()  — overdispersed count noise
        build_homeostatic_cells()      — 7 digit classes as anchor rooms
        build_perturbation_cells()     — 5 pert types x 3 doses
        build_ood_cells()              — held-out OOD perturbation
        run_sanity_checks()            — verifies key dataset properties
        save_dataset()                 — saves .npz + metadata JSON

    Usage:
        python build_proxy_dataset.py                      # full dataset
        python build_proxy_dataset.py --no_nb_noise        # fast (dev)
        python build_proxy_dataset.py --n_per_class 200    # tiny (smoke test)

### geo_jepa_mnist.py
    Training script. Reads from proxy_data/ (run build_proxy_dataset.py first).

    Key classes:
        GeneInteractionPrior       — Stage 1: sparse attention over genes
        ReduNetLayer               — one unrolled MCR² gradient step
        ReduNetEncoder             — L layers of ReduNetLayer
        GeoJEPAEncoder             — Stage 1 + Stage 2 combined
        FateConditionalPredictor   — MLP predictor gated by Pi
        CellDataset                — PyTorch Dataset wrapper

    Key functions:
        build_proxy_dataset()       — dataset construction (embedded version)
        gene_set_mask()             — context/target gene partitioning
        coding_rate()               — R(Z, ε) via Cholesky log-det
        mcr2_loss()                 — ΔR = R(Z) - Σ_k R(Z_k)
        ema_update()                — target encoder EMA step
        train()                     — main training loop
        evaluate_discrimination()   — computes Discrimination Ratio

    Default hyperparameters:
        gene_dim        = 2000
        embed_dim       = 128
        n_classes       = 7       (one per room digit)
        n_heads         = 4       (attention)
        max_genes       = 200     (top-k expressed genes per cell)
        n_layers        = 6       (ReduNet gradient steps)
        predictor_hidden = 256
        context_ratio   = 0.75
        ema_momentum    = 0.999
        mcr_weight      = 0.1
        eps_sq          = 0.5
        batch_size      = 128
        epochs          = 30
        lr              = 3e-4

    Usage:
        python geo_jepa_mnist.py                           # defaults
        python geo_jepa_mnist.py --epochs 5 --n_per_class 200  # smoke test
        python geo_jepa_mnist.py --n_layers 8 --embed_dim 256  # larger run

---

## KNOWN ISSUES AND TODOS (ordered by priority)

    [P1] Phase A / Phase B are not separated.
         Current prototype trains on homeostatic and perturbed cells jointly.
         True dual-phase curriculum requires:
           Phase A: train on homeostatic_cells only until convergence
           Phase B: introduce perturbation_cells with EMA anchor loss
         EMA anchor: L_anchor = ||sg(z_home_ema) - z_home_online||²

    [P1] Perturbation conditioning is missing.
         The predictor receives Pi (subspace assignment) but NOT the
         perturbation identity or its embedding. For real scRNA-seq, the
         perturbation embedding (from GO graph or ESM-2) is the mechanism
         that enables OOD generalization. In the proxy, the perturbation
         can be encoded as the delta between digit means (a 784-dim vector
         projected through W), providing structure for OOD generalization.

    [P2] ReduNetLayer._woodbury_mv has two code paths (n<d and n>=d).
         The n>=d path uses Woodbury + Cholesky correctly but the algebra
         should be verified with a unit test comparing to a brute-force
         matrix inverse on a small synthetic example.

    [P2] Online soft clustering in estimate_assignments() uses random
         centroid initialization each forward pass. This causes noisy Pi
         estimates on small batches. Consider: maintain running centroids
         as a buffer updated with EMA, similar to k-means with momentum.

    [P3] evaluate_discrimination() uses L2 distance in embedding space.
         Should also compute cosine distance for comparison, since
         embeddings are L2-normalized.

    [P3] No visualization. Add a UMAP plot of homeostatic embeddings after
         Phase A training to verify that the 7 rooms are orthogonally
         separated before starting Phase B.

    [P3] NB noise in build_proxy_dataset.py loops over genes (slow).
         Vectorize using numpy broadcasting or scipy.

---

## MATHEMATICAL BACKGROUND

### MCR² (Maximal Coding Rate Reduction)  — Ma et al.
    Objective: ΔR(Z, Π) = R(Z) - Σ_k (n_k/n) R(Z_k)

    Coding rate:
        R(Z, ε) = (d/2) log det(I + d/(nε²) ZZ^T)

    Maximizing ΔR simultaneously:
        - Expands the full representation to fill ambient space  [R(Z) term]
        - Compresses each class into a low-dimensional subspace  [R(Z_k) term]
        - Separates classes into orthogonal subspaces            [emerges from both]

    The ReduNet unrolls gradient ascent on ΔR into a feedforward network.
    Reference: Ma et al. "Principles and Practice of Deep Representation
    Learning." Also: Wang et al. 2021 (ReduNet paper, JMLR).

### JEPA (Joint-Embedding Predictive Architecture)  — LeCun 2022
    Instead of reconstructing x_target from x_context (generative),
    predict EMBEDDING of x_target from EMBEDDING of x_context:

        ẑ_T = g(E_θ(x_C))    trained to match    sg(E_φ(x_T))

    where E_φ is a slow-moving EMA target encoder.
    Advantage: forces learning of abstract structure, not surface statistics.
    Reference: Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023.
    Applied to scRNA-seq: Litman et al. "GeneJEPA" bioRxiv 2025.

### Memory Theory Connection  — Buchanan, Pai, Wang, Ma
    The MCR² subspace structure enables associative recall: a partial
    observation (cell with 80% dropout) is recovered by projecting onto
    the nearest subspace. The MNAR masking schedule can be set from
    the theory's bounds on tolerable masking fraction.
    Reference: "A Mathematical Theory of Memory" (Buchanan, Pai, Wang, Ma).

---

## RELEVANT PAPERS (with one-line relevance notes)

    [1] Ahlmann-Eltze, Huber, Anders. "Deep-learning-based gene perturbation
        effect prediction does not yet outperform simple linear baselines."
        Nature Methods 2025. DOI: 10.1038/s41592-025-02772-6
        -> Defines the floor: simple mean baseline beats all current DL models.

    [2] Roohani, Huang, Leskovec. "Predicting transcriptional outcomes of
        novel multigene perturbations with GEARS." Nature Biotechnology 2024.
        DOI: 10.1038/s41587-023-01905-6
        -> Structured perturbation embeddings (GO graph) enable OOD generalization.

    [3] Lotfollahi et al. "Predicting cellular responses to complex
        perturbations in high-throughput screens." (CPA) Mol Sys Bio 2023.
        DOI: 10.15252/msb.202211517
        -> Direct ancestor of the perturbation-as-vector-addition approach.

    [4] Klein et al. "CellFlow enables generative single-cell phenotype
        modeling with flow matching." bioRxiv 2025.
        DOI: 10.1101/2025.04.11.648220
        -> Competing approach using flow matching; handles distributional outputs.

    [5] Litman et al. "GeneJEPA: A Predictive World Model of the
        Transcriptome." bioRxiv 2025. DOI: 10.1101/2025.10.14.682378
        -> Direct JEPA-for-scRNA-seq implementation; Perceiver encoder.

    [6] VCC Commentary: "Virtual Cell Challenge: Toward a Turing test for
        the virtual cell." Cell 2025. DOI: 10.1016/j.cell.2025.00675
        -> Benchmark definition, metrics (DES, PDS, MAE), H1 hESC test set.

    [7] Norman et al. "Exploring genetic interaction manifolds constructed
        from rich single-cell phenotypes." Science 2019.
        -> Standard held-out perturbation benchmark (K562 combinatorial CRISPRi).

---

## QUICK START FOR THIS SESSION

    # 1. Install dependencies
    pip install torch scikit-learn scipy numpy

    # 2. Generate the proxy dataset
    python build_proxy_dataset.py --no_nb_noise --n_per_class 200
    # (add --no_nb_noise for fast iteration; remove for realistic noise)

    # 3. Smoke test the training loop
    python geo_jepa_mnist.py --epochs 5 --n_per_class 200

    # 4. Full training run
    python build_proxy_dataset.py
    python geo_jepa_mnist.py --epochs 30

    # 5. Watch for:
    #    - pred_loss decreasing      (JEPA prediction learning)
    #    - mcr_loss becoming less negative  (subspace structure forming)
    #    - Discrimination Ratio > 2.0 at end  (Linearization Trap avoided)

---

## SESSION TASK

[REPLACE THIS SECTION WITH YOUR SPECIFIC TASK FOR THIS SESSION]

Example tasks:
- "Implement the Phase A / Phase B curriculum split (P1 above)"
- "Add perturbation conditioning to the predictor (P1 above)"
- "Write a unit test for ReduNetLayer._woodbury_mv (P2 above)"
- "Add running centroid EMA to estimate_assignments (P2 above)"
- "Add UMAP visualization of Phase A embeddings (P3 above)"
- "Profile the training loop and identify MPS bottlenecks"
- "Implement the OOD evaluation using perturbation embedding similarity"
