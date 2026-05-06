# Manuscript Plan: FC-GNN
## *Fuzzy-Conformal Graph Neural Networks with Distribution-Free Coverage Guarantees for Cybersecurity Anomaly Detection*

**Target Journal:** International Journal of Machine Learning and Cybernetics (IJMLC), Springer  
**ISSN:** 1868-8071 | **Impact Factor (JCR 2025):** 3.80 (Q2) | **CiteScore:** 7.7  
**Contribution type:** New algorithm with theoretical guarantees + empirical benchmarks

---

## 1. Problem Statement

### 1.1 Motivation

Network intrusion detection, botnet identification, and IoT malware analysis are canonical cybersecurity tasks that operate over inherently *graph-structured* data: traffic flows between hosts form edges, devices form nodes, and attack campaigns manifest as anomalous subgraph patterns. Graph Neural Networks (GNNs) have emerged as powerful classifiers for these tasks—achieving high accuracy on benchmarks such as CIC-IDS-2017, UNSW-NB15, and NF-BoT-IoT—yet they suffer from two critical, simultaneously unresolved weaknesses:

1. **Epistemic uncertainty from noisy and ambiguous features.** Network telemetry is inherently imprecise: packet-level features exhibit quantization noise, flow statistics are aggregated over variable windows, and labeling of attack vs. benign traffic is often soft or partially overlapping. Standard GNN classifiers produce crisp binary decisions that discard this inherent fuzziness, leading to brittle predictions and inflated false-alarm rates in production SOC deployments.

2. **Absence of formal coverage guarantees.** Neither standard GNNs nor fuzzy-neural extensions provide distribution-free, finite-sample guarantees on their prediction sets. A security analyst receiving an alert has no principled bound on the probability that the true attack class lies within the model's output — making risk quantification impossible for compliance-sensitive (e.g., GDPR, NIS2) environments.

### 1.2 Research Gaps

Despite parallel progress in (a) conformal prediction for GNNs and (b) fuzzy GNN architectures, their explicit fusion remains an open problem:

| Paradigm | Coverage Guarantee | Fuzzy Aggregation | Cybersecurity Benchmarks |
|---|---|---|---|
| CF-GNN (NeurIPS 2023) | ✅ | ❌ | ❌ |
| RR-GNN (UAI 2025) | ✅ | ❌ | ❌ |
| FL-GNN (ICLR 2024) | ❌ | ✅ | ❌ |
| FGAT / MFGAT (2024) | ❌ | ✅ | ❌ |
| GraphIDS (NeurIPS 2025) | ❌ | ❌ | ✅ |
| **FC-GNN (ours)** | **✅** | **✅** | **✅** |

No published work as of May 2026 simultaneously provides (i) fuzzy-rule-based aggregation for uncertainty-aware GNN message passing, (ii) distribution-free prediction sets via conformal prediction, and (iii) empirical validation on cybersecurity datasets.

### 1.3 Research Questions

- **RQ1.** Can fuzzy-rule-based message passing replace crisp GNN aggregation to produce soft, uncertainty-aware node embeddings without sacrificing predictive accuracy on imbalanced cybersecurity graphs?
- **RQ2.** Can Graph-Structured Mondrian Conformal Prediction be extended to wrap a fuzzy GNN to provide cluster-conditional, finite-sample coverage guarantees for multi-class intrusion detection?
- **RQ3.** Does the resulting Fuzzy-Conformal GNN (FC-GNN) yield tighter prediction sets and lower false-alarm rates than existing CP-GNN and fuzzy-GNN baselines across multiple cybersecurity benchmark datasets?
- **RQ4.** Does the fuzzy rule base provide human-interpretable explanations of uncertain predictions that assist SOC analyst workflows?

### 1.4 Contributions

1. **FC-GNN Architecture.** A novel graph neural network in which each message-passing layer incorporates a fuzzy-rule aggregation module (extending FL-GNN's Cartesian-product rule space) that outputs a fuzzy membership distribution over classes per node, rather than a crisp softmax logit.
2. **Fuzzy Mondrian Conformal Prediction (FMCP).** A principled extension of Graph-Structured Mondrian CP (RR-GNN, UAI 2025) to fuzzy nonconformity scores, where the nonconformity measure is defined via a Sugeno-type fuzzy integral over the membership distribution, producing calibrated prediction sets with per-cluster conditional coverage guarantees.
3. **Coverage Theorem.** A finite-sample theorem proving that FC-GNN's prediction sets achieve at least (1−α) marginal coverage and conditional coverage within graph communities, under the fuzzy-extended permutation-invariance condition.
4. **Cybersecurity Benchmarks.** Systematic evaluation on six publicly available cybersecurity datasets covering intrusion detection (CIC-IDS-2017, UNSW-NB15), botnet detection (ISCX Botnet, CTU-13), and IoT malware (N-BaIoT, NF-BoT-IoT), with temporal train/calibration/test splits to assess behavior under concept drift.
5. **Interpretability Analysis.** Extraction and visualization of top-K fuzzy rules firing per attack family, providing human-readable IF-THEN explanations of model uncertainty aligned with MITRE ATT&CK categories.

---

## 2. Literature Review

### 2.1 Conformal Prediction — Foundations

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 1 | A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | Angelopoulos & Bates | arXiv 2107.07511 | 2021 | Standard tutorial; marginal vs. conditional coverage definitions |
| 2 | Conformal Prediction Beyond Exchangeability | Barber, Candès, Tibshirani, Wager | *Annals of Statistics* 51(2) | 2023 | Weighted CP and covariate-shift robustness |
| 3 | Conformal Prediction: A Data Perspective | Zhou et al. | *ACM Computing Surveys* (10.1145/3736575) | 2025 | Comprehensive survey with graph-data sub-section |
| 4 | CONFIDE: CONformal Free Inference for Distribution-Free Estimation in Causal Competing Risks | Dang, Q.-V. | *Mathematics* (MDPI) 14(2):383 | 2026 | Authors' own CP work; causal competing-risk prediction sets |
| 5 | Conformal Prediction in the Intrusion Detection Problem | Dang, Q.-V. | *J. Info. Assurance & Security* 18(1) | 2023 | Authors' own CP-IDS baseline |
| 6 | Kernel Methods for Conformal Prediction to Detect Botnets | Dang, Q.-V. & Pham, T.-H. | AITA 2023 (Springer LNNS 843) | 2024 | Authors' own kernel-CP botnet detection |

### 2.2 Conformal Prediction for Graph Neural Networks

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 7 | Uncertainty Quantification over Graph with Conformalized Graph Neural Networks (CF-GNN) | Huang, Jin, Candès, Leskovec | NeurIPS 2023 (arXiv 2305.14535) | 2023 | De-facto baseline; topology-aware correction; permutation-invariance condition |
| 8 | Conformal Prediction Sets for Graph Neural Networks (DAPS) | Zargarbashi, Antonelli, Bojchevski | ICML 2023 (PMLR 202) | 2023 | Diffusion of conformity scores via homophily |
| 9 | Distribution-Free Prediction Sets for Node Classification | Clarkson | ICML 2023 (arXiv 2211.14555) | 2023 | Inductive node CP with structure-aware weighting |
| 10 | Similarity-Navigated Conformal Prediction for GNNs (SNAPS) | Song, Huang, Jiang, Zhang, Li, Wang | NeurIPS 2024 (arXiv 2405.14303) | 2024 | Feature-similar neighbor score aggregation; singleton-hit improvement |
| 11 | Conformalized Link Prediction on Graph Neural Networks | Zhao, Kang, Cheng | KDD 2024 (ACM 10.1145/3637528.3672061) | 2024 | First model-agnostic CP for GNN link prediction |
| 12 | Conformal Inductive Graph Neural Networks | — | ICLR 2024 (OpenReview homn1jOKI5) | 2024 | Node- and edge-exchangeable inductive CP |
| 13 | Valid Conformal Prediction for Dynamic GNNs | — | ICLR 2025 (arXiv 2405.19230) | 2025 | Tensor-unfolding for exchangeability recovery in temporal GNNs |
| 14 | Residual Reweighted Conformal Prediction for GNNs (RR-GNN) | Zhang, Bao, Zhou, Colombo, Cheng, Luo | UAI 2025 (PMLR 286:4982–4999; arXiv 2506.07854) | 2025 | Graph-Structured Mondrian CP; residual reweighting; cross-training protocol |
| 15 | Conformal Prediction for Federated GNNs with Missing Neighbor Information | Akgül, Kannan, Prasanna | UAI 2025 (PMLR 286:45–63) | 2025 | Federated CP-GNN under partial exchangeability |
| 16 | Non-exchangeable Conformal Prediction for Temporal GNNs (NCPNet) | Wang, Kang, Yan, Kulkarni, Zhou | KDD 2025 (arXiv 2507.02151) | 2025 | Handling temporal dependency that violates exchangeability |
| 17 | Enhancing Trustworthiness of GNNs with Rank-Based Conformal Training | — | AAAI 2025 (arXiv 2501.02767) | 2025 | CP integrated into GNN training loop |
| 18 | RoCP-GNN: Robust Conformal Prediction for GNNs | Akansha | arXiv 2408.13825 | 2024 | CP robustness under distributional shift |
| 19 | CRC-SGAD: Conformal Risk Control for Supervised Graph Anomaly Detection | Bai et al. | arXiv 2504.02248 | 2025 | CP extended to graph anomaly detection with risk control |
| 20 | Fuzzy Prediction Sets: Conformal Prediction with E-values | Koning & van Meer | arXiv 2509.13130 | 2026 | Fuzzy CP via e-values (non-graph; key theoretical bridge) |

### 2.3 Fuzzy Graph Neural Networks

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 21 | FL-GNN: A Fuzzy-Logic Graph Neural Network | — | ICLR 2024 (OpenReview RTLjdy6Ntk) | 2024 | Cartesian-product fuzzy rule base + GNN message passing; interpretable rules |
| 22 | Enhancing Link Prediction with Fuzzy Graph Attention Networks and Dynamic Negative Sampling (FGAT) | Xing, Xue et al. | arXiv 2411.07482 | 2024 | Fuzzy rough-set attention + dynamic negative sampling |
| 23 | Multi-view Fuzzy Graph Attention Networks for Enhanced Graph Learning (MFGAT) | — | arXiv 2412.17271 | 2024 | Multi-view fuzzy GAT + learnable global pooling |
| 24 | Fuzzy Graph Neural Networks: A Comprehensive Review of Uncertainty-Aware Graph Learning | Tran, Tong, Nguyen, Nguyen | *EAI Endorsed Trans. CASA* vol. 10 | 2025 | First dedicated FGNN survey; gap identification |
| 25 | An Integration of Fuzzy-Enhanced Heterophilic Graph Representation Learning (FHGE) | — | *Cybernetics and Systems* (Taylor & Francis) | 2025 | Bidirectional fuzzy attention + GIN for heterophilic link prediction |
| 26 | An Integrated Fuzzy Neural Network and Topological Data Analysis (FTPG) | Pham et al. | *Molecular Informatics* (Wiley, 10.1002/minf.202400335) | 2025 | FNN + persistent-homology TDA for graph property prediction |
| 27 | FireGNN: Neuro-Symbolic GNNs with Trainable Fuzzy Rules | — | arXiv 2509.10510 | 2025 | Trainable IF-THEN fuzzy rules in neuro-symbolic GNN |
| 28 | Explainable Fuzzy GNNs for Leak Detection in Water Distribution Networks (FGENConv) | Khaled et al. | arXiv 2601.03062 | 2026 | Fuzzy logic + mutual info on GENConv; rule-based explanations |
| 29 | Fuzzy Embedding to Detect Intrusion in Software-Defined Networks | Dang, Q.-V. | Springer LNNS (Intelligent and Fuzzy Systems, 10.1007/978-3-031-67195-1_78) | 2024 | Authors' own fuzzy-embedding NIDS baseline |
| 30 | Improving GNNs by Learning Continuous Edge Directions (CoED-GNN) | — | ICLR 2025 (arXiv 2410.14109) | 2025 | Fuzzy Laplacian for continuous in/out flow direction |

### 2.4 GNNs for Cybersecurity (Intrusion, Botnet, IoT Malware)

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 31 | GraphIDS: Self-Supervised GNN for Network Intrusion Detection | Guerra, Chapuis, Duc, Mozharovskyi, Nguyen | NeurIPS 2025 | 2025 | E-GraphSAGE + transformer AE; 99.98% PR-AUC |
| 32 | GNN-IDS: Graph Neural Network based Intrusion Detection System | Sun et al. | ARES 2024 (ACM 10.1145/3664476.3664515) | 2024 | Static attack graph + dynamic telemetry; uncertainty evaluation |
| 33 | XG-NID: Dual-Modality NIDS using Heterogeneous GNN and LLM | — | arXiv 2408.16021 | 2024 | Heterogeneous GNN + LLM dual modality |
| 34 | BS-GAT: Behavior Similarity Based GAT for Network Intrusion Detection | — | arXiv 2304.07226 | 2025 | Behavior-similarity GAT for NIDS |
| 35 | CAGN-GAT Fusion: Hybrid Contrastive Attentive GNN for NIDS | — | arXiv 2503.00961 | 2025 | Contrastive + attentive GNN fusion |
| 36 | EL-GNN: Continual-Learning GNN for Task-Incremental IDS | — | *MDPI Electronics* 14(14):2756 | 2025 | Elastic weight consolidation GNN for evolving threats |
| 37 | Survey on GNNs for IDS: Methods, Trends, Challenges | — | *Computers & Security* 141 | 2024 | Comprehensive GNN-IDS survey |
| 38 | Detecting IoT Malware Using Federated Learning | Dang, Q.-V. & Pham, T.-H. | Springer Comm. CIS | 2024 | Authors' own federated IoT-malware baseline |
| 39 | Detecting Intrusion in WiFi Network Using GNNs | Dang, Q.-V. & Nguyen, T.-L. | ICCCES 2022 (Springer 2023) | 2023 | Authors' own GNN-NIDS baseline |
| 40 | A Survey of Heterogeneous GNNs for Cybersecurity Anomaly Detection | — | arXiv 2510.26307 | 2025 | Survey of hetero-GNN anomaly detection |

### 2.5 Conformal Prediction in Cybersecurity

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 41 | Comprehensive Botnet Detection with Conformal Layers | Issac et al. | *Information Fusion* (arXiv 2409.00667) | 2024 | CP rejection layer; 58% incorrect-prediction rejection on ISCX |
| 42 | Conformal Prediction for Labelling and Updating Online Models under Concept Drift in Cybersecurity | — | *J. Information Security and Applications* (S2214212625001577) | 2025 | CP pseudo-labels inconsistent under heavy drift; key caveat |
| 43 | DA-MBA: Denoising Adaptive Multi-Branch Architecture with Open-Set Conformal Calibration | — | *J. Cybersecurity & Privacy* 6(1):26 | 2026 | Open-set CP for zero-day IIoT detection |
| 44 | Conformal ML for Reliable Anomaly Detection in Industrial Cyber-Physical Systems | — | *Reliability Engineering & System Safety* (S0951832026002334) | 2026 | Sliding-window temporal quantile CP; FAR guarantees |
| 45 | Context-Aware Online Conformal Anomaly Detection (C-PP-COAD) | — | arXiv 2505.01783 | 2025 | Prediction-powered CP under scarce calibration data |

### 2.6 Fuzzy Methods in Cybersecurity

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 46 | A Cooperative IDS for IoT using Fuzzy Logic and CNN Ensemble | Qiu, Shi, Fan | *Scientific Reports* 15:15934 | 2025 | NSL-KDD 99.72%, NSW-NB15 98.36%; fuzzy+CNN ensemble |
| 47 | Fuzzy-Rule-Based Optimized Hybrid Deep Learning for SDN-Enabled IoT NIDS | — | *Computers & Security* (S0167404825000616) | 2025 | Hunger-Games-optimized fuzzy-rule DL for SDN |
| 48 | GFS-GAN: Genetic Fuzzy Systems with Adversarial Training for IDS | — | *Computers & Security* (S0167404825002664) | 2025 | Evolved fuzzy rules + adversarial augmentation |
| 49 | Enhancing Security using Fuzzy Graph Theory | — | *Scientific Reports* | 2025 | Fuzzy graph theory for access control; 95% accuracy |
| 50 | Fuzzy-Based Signature IDS with Clustering | Ahmed et al. | *Scientific Reports* (s41598-025-85866-7) | 2025 | Signature-based IDS with fuzzy clustering |

---

## 3. Proposed Algorithm: FC-GNN

### 3.1 Architecture Overview

```
Input Graph G = (V, E, X)                         [nodes V, edges E, feature matrix X]
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  FUZZY MESSAGE-PASSING LAYERS  (L layers)                       │
│                                                                  │
│  For each node v at layer ℓ:                                    │
│  1. Neighbor aggregation → fuzzy membership scores              │
│     μ_k(h_u^{ℓ})  for each rule k ∈ {1,...,K}                  │
│  2. Fuzzy rule firing strength:                                  │
│     τ_k(v) = T-norm( μ_k(h_u^{ℓ})  ∀ u ∈ N(v) )               │
│  3. Defuzzification → updated embedding h_v^{ℓ+1}              │
│     h_v^{ℓ+1} = Σ_k  [ τ_k(v) · w_k ] / Σ_k τ_k(v)           │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  FUZZY MEMBERSHIP OUTPUT HEAD                                    │
│  p_v = Softmax( MLP(h_v^L) )                                    │
│  → fuzzy class membership vector μ_v ∈ [0,1]^C                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  FUZZY MONDRIAN CONFORMAL PREDICTION  (post-hoc calibration)    │
│                                                                  │
│  1. Partition calibration nodes into communities Q_1,...,Q_M   │
│     via graph clustering (Louvain / spectral)                   │
│  2. Fuzzy nonconformity score for node v in class c:            │
│     s_v = 1 − S_λ(μ_v, y_v)    [Sugeno fuzzy integral]         │
│  3. Per-community quantile:                                      │
│     q̂_{Q_m} = ⌈(|Q_m|+1)(1−α)⌉-th smallest s in Q_m          │
│  4. Fuzzy prediction set at test node v ∈ Q_m:                  │
│     C_α(v) = { c : 1 − S_λ(μ_v, c) ≤ q̂_{Q_m} }               │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
Output: Calibrated attack-class prediction set C_α(v)
        + Fired fuzzy rules for interpretability
```

### 3.2 Key Components

#### 3.2.1 Fuzzy Message-Passing Layer (FMPL)

Replaces the standard GNN aggregation function (mean / sum / max) with a fuzzy-rule-based combiner:

- **Membership functions.** For each input feature dimension d, define K_d triangular or Gaussian membership functions (learnable centers and widths).
- **Rule construction.** Following FL-GNN (ICLR 2024), define M^D rules in the Cartesian product of K_d membership functions per dimension, pruned by relevance to ≤ K active rules.
- **Aggregation.** For node v, aggregate neighbor embeddings via weighted Mamdani-style rule firing. The T-norm uses the product T-norm (differentiable; suitable for gradient-based learning).
- **Defuzzification.** Centroid defuzzification outputs a continuous real-valued embedding h_v^{ℓ+1}.

#### 3.2.2 Fuzzy Nonconformity Score

Replace the standard softmax-based score ( s_v = 1 − p̂_v(y_v) ) with a **Sugeno fuzzy integral**:

```
S_λ(μ_v, y_v) = sup_{A ⊆ C} [ min( min_{c∈A} μ_v(c),  g_λ(A) ) ]
```

where g_λ is a fuzzy measure (λ-measure) learned from calibration data, encoding how subsets of attack classes co-occur. This assigns a fuzzily-weighted confidence that the true class y_v is consistent with the membership distribution μ_v.

#### 3.2.3 Graph-Structured Mondrian CP (FMCP)

Extends RR-GNN's Mondrian partitioning to the fuzzy setting:

- Communities Q_m defined over the *calibration* graph via Louvain clustering.
- Per-community conditional coverage guarantee:

```
P( y_v ∈ C_α(v) | v ∈ Q_m ) ≥ 1 − α − δ(|Q_m|)
```

where δ(|Q_m|) is a finite-sample slack term derived from a Hoeffding-type argument over fuzzy-score distributions.

#### 3.2.4 Coverage Theorem (informal statement)

**Theorem (FC-GNN Coverage).** *Let G be a graph satisfying the fuzzy-extended permutation invariance condition (Definition 3.1). Let C_α(v) be the FC-GNN Mondrian prediction set for node v in community Q_m calibrated at level α. Then for any test node v ∈ Q_m drawn exchangeably within Q_m:*

```
P( y_v ∈ C_α(v) ) ≥ 1 − α
```

*Furthermore, if the community membership assignment is consistent (Assumption A2), FC-GNN achieves conditional coverage within Q_m up to a finite-sample slack δ(|Q_m|) = O(|Q_m|^{-1/2}).*

### 3.3 Training Procedure

```
Algorithm FC-GNN Training
─────────────────────────────────────────────────────────────────
Input:  Graph G, labels Y_train, calibration set V_cal, α ∈ (0,1)
Output: Trained FC-GNN + community-conditional quantiles q̂_{Q_m}

Phase 1 — Fuzzy GNN Training
  1. Initialize fuzzy membership centers and widths randomly
  2. Train FMPL layers + MLP head via cross-entropy on V_train
     (with cross-training split to avoid calibration leakage)
  3. Regularize firing strengths via L1 on τ_k to prune inactive rules

Phase 2 — Fuzzy Mondrian CP Calibration
  4. Forward pass on V_cal → obtain μ_v for all v ∈ V_cal
  5. Partition V_cal into communities {Q_1,...,Q_M} via Louvain
  6. Compute fuzzy nonconformity scores s_v = 1 − S_λ(μ_v, y_v)
  7. For each Q_m: q̂_{Q_m} = ⌈(|Q_m|+1)(1−α)⌉-th order statistic

Phase 3 — Inference
  8. For test node v: assign to nearest community Q_m
  9. Compute C_α(v) = { c : 1 − S_λ(μ_v, c) ≤ q̂_{Q_m} }
 10. Extract top-K fired rules for IF-THEN explanation
─────────────────────────────────────────────────────────────────
```

### 3.4 Baselines for Comparison

| Baseline | Type | Coverage Guarantee | Fuzzy |
|---|---|---|---|
| CF-GNN (Huang et al., NeurIPS 2023) | CP-GNN | ✅ marginal | ❌ |
| DAPS (Zargarbashi et al., ICML 2023) | CP-GNN | ✅ marginal | ❌ |
| RR-GNN (Zhang et al., UAI 2025) | CP-GNN | ✅ Mondrian-conditional | ❌ |
| SNAPS (Song et al., NeurIPS 2024) | CP-GNN | ✅ marginal | ❌ |
| FL-GNN (ICLR 2024) | Fuzzy-GNN | ❌ | ✅ |
| FGAT (Xing et al., 2024) | Fuzzy-GNN | ❌ | ✅ |
| GraphIDS (Guerra et al., NeurIPS 2025) | GNN-IDS | ❌ | ❌ |
| GNN-IDS (Sun et al., ARES 2024) | GNN-IDS | ❌ | ❌ |
| Fuzzy-Rule SDN-IDS (Computers & Security 2025) | Fuzzy-IDS | ❌ | ✅ |
| **FC-GNN (ours)** | **Fuzzy-CP-GNN** | **✅ Mondrian-conditional** | **✅** |

---

## 4. Evaluation Datasets

### 4.1 Network Intrusion Detection

| # | Dataset | Description | Classes | Size | Link |
|---|---|---|---|---|---|
| D1 | **CIC-IDS-2017** | Canadian Institute for Cybersecurity, 2017. Full packet captures of benign + 14 attack types (DoS, DDoS, Brute Force, Web Attacks, Infiltration, Botnet). Standard 5-tuple flow features. | 15 (benign + 14 attacks) | ~2.8M flows | https://www.unb.ca/cic/datasets/ids-2017.html |
| D2 | **UNSW-NB15** | UNSW Canberra, 2015. 49 features from network traffic; 9 attack categories + normal. Widely used IDS benchmark with feature diversity. | 10 | ~2.5M records | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| D3 | **NF-BoT-IoT** (NetFlow version) | NetFlow-extracted version of BoT-IoT, 2020. Covers DDoS, DoS, Reconnaissance, Theft in IoT environments. Graph-compatible via IP-to-IP flow aggregation. | 5 | ~4.6M flows | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ |

### 4.2 Botnet Detection

| # | Dataset | Description | Classes | Size | Link |
|---|---|---|---|---|---|
| D4 | **ISCX Botnet 2014** | University of New Brunswick. Network traces from IRC-, P2P-, HTTP-based botnets (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay). | 7 botnets + benign | ~2.7GB PCAP / ~650K flows | https://www.unb.ca/cic/datasets/botnet.html |
| D5 | **CTU-13** | Czech Technical University, 2011. 13 labeled botnet scenarios (Neris, Rbot, Menti, Sogou, Murlo etc.) in real traffic. Canonical botnet graph benchmark. | 13 scenarios | ~1.7M flows | https://www.stratosphereips.org/datasets-ctu13 |

### 4.3 IoT Malware Detection

| # | Dataset | Description | Classes | Size | Link |
|---|---|---|---|---|---|
| D6 | **N-BaIoT** | Ben-Gurion University, 2018. Statistical feature vectors from 9 commercial IoT devices infected with Mirai and BASHLITE variants. Device-level graph via ARP/traffic correlation. | 10 (9 attack subtypes + benign) | ~7.06M samples | https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot |
| D7 | **ToN-IoT** | UNSW + University of New South Wales, 2020. Multi-source IoT telemetry (network + system + OS logs). 9 attack types in heterogeneous IoT environment. | 10 | ~461K network rows | https://research.unsw.edu.au/projects/toniot-datasets |

### 4.4 Graph Construction Protocol

Each dataset will be converted to a graph as follows:
- **Nodes:** unique IP addresses or device identifiers
- **Edges:** directional flow between source and destination IP/port pairs, aggregated within a time window W (e.g., W = 60s)
- **Node features:** aggregated flow statistics per IP (mean/std of packet sizes, inter-arrival times, byte ratios, active/idle times)
- **Edge features:** individual flow-level features (protocol, port, byte count, duration)
- **Labels:** per-node attack category (majority label within window), enabling node-classification formulation
- **Temporal splits:** 60% train / 20% calibration / 20% test in *chronological order* (not random) to simulate concept drift realistic in production

---

## 5. Evaluation Metrics

### 5.1 Predictive Performance (Point Prediction)

| Metric | Formula | Rationale |
|---|---|---|
| **Accuracy** | (TP + TN) / N | Standard baseline metric |
| **Macro-F1** | Mean F1 across classes | Handles class imbalance (intrusion datasets are heavily skewed) |
| **Area Under PR Curve (AUPRC)** | ∫ P(r) dr | Preferred over AUROC for imbalanced cybersecurity data |
| **False Alarm Rate (FAR)** | FP / (FP + TN) | Critical for SOC deployment; false alarms waste analyst time |
| **Matthews Correlation Coefficient (MCC)** | Standard formula | Single balanced metric for binary and multi-class |

### 5.2 Conformal Prediction Quality

| Metric | Formula | Rationale |
|---|---|---|
| **Empirical Coverage** | \|{v : y_v ∈ C_α(v)}\| / \|V_test\| | Must be ≥ (1−α) to validate the coverage guarantee |
| **Average Prediction Set Size (APSS)** | Mean \|C_α(v)\| over V_test | Efficiency: smaller = more informative; compared at fixed α = 0.1 |
| **Singleton Hit Proportion (SHP)** | \|{v : \|C_α(v)\| = 1}\| / \|V_test\| | Fraction of nodes with a single-class prediction; higher = more decisive |
| **Conditional Coverage (per community Q_m)** | \|{v ∈ Q_m : y_v ∈ C_α(v)}\| / \|Q_m\| | Validates Mondrian-conditional guarantee across attack-family clusters |
| **Coverage Gap** | max_m \| P(y∈C_α \| Q_m) − (1−α) \| | Worst-case deviation from target coverage per community |
| **Conformal Efficiency Gain** | (APSS_baseline − APSS_FC-GNN) / APSS_baseline × 100% | Relative set-size reduction vs. CF-GNN / RR-GNN at same α |

### 5.3 Interpretability Metrics

| Metric | Description |
|---|---|
| **Rule Fidelity** | Proportion of test nodes whose top-1 fired rule is consistent with ground-truth attack family (MITRE ATT&CK mapping) |
| **Rule Complexity** | Mean number of antecedents per active rule (lower = more human-readable) |
| **Rule Coverage** | Proportion of test nodes covered by the top-K rules (K = 10) |
| **Explanation Stability** | Mean Jaccard similarity between fired rule sets for similar inputs (robustness of explanations) |

### 5.4 Concept Drift Robustness

| Metric | Description |
|---|---|
| **Temporal Coverage Drop** | Δ Empirical Coverage between early and late test splits |
| **FAR under Drift** | FAR computed on the final chronological test quarter (most drifted) |
| **CP Recalibration Frequency** | Number of sliding-window recalibrations needed to maintain target coverage |

### 5.5 Significance Testing

- **Wilcoxon signed-rank test** (non-parametric) comparing F1 / APSS pairs across 5-fold cross-validation runs
- **Friedman + Nemenyi post-hoc test** for ranking across multiple baselines and datasets
- Report **p < 0.05** significance for all primary claims

---

## 6. Paper Structure (Planned)

| Section | Content | Target Length |
|---|---|---|
| Abstract | Problem, method, results, impact | 250 words |
| 1. Introduction | Motivation, gaps, contributions, paper structure | 1.5 pages |
| 2. Related Work | CP-GNNs, Fuzzy-GNNs, CP-Cybersecurity, Fuzzy-Cybersecurity | 2.5 pages |
| 3. Preliminaries | CP fundamentals, GNN formalism, fuzzy sets / integrals | 1.5 pages |
| 4. FC-GNN | Fuzzy MP layer, FMCP, coverage theorem + proof sketch | 3.5 pages |
| 5. Experiments | Datasets, baselines, metrics, results, ablations | 3.5 pages |
| 6. Interpretability Analysis | Rule extraction, MITRE ATT&CK mapping, case study | 1.5 pages |
| 7. Discussion | Limitations (concept drift, exchangeability, scalability) | 0.75 pages |
| 8. Conclusion | Summary, future work (federated FC-GNN, temporal extension) | 0.5 pages |
| References | ~50 citations | — |

---

## 7. Timeline (Indicative)

| Milestone | Target |
|---|---|
| Literature review finalized | Week 1 |
| Formal algorithm + theorem write-up | Week 2 |
| Code implementation (PyTorch Geometric) | Weeks 3–4 |
| Experiments on D1–D3 (intrusion) | Week 5 |
| Experiments on D4–D7 (botnet + IoT) | Week 6 |
| Ablation + interpretability analysis | Week 7 |
| Full draft manuscript | Week 8 |
| Internal review + revision | Week 9 |
| Submission to IJMLC | Week 10 |
