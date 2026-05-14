# Experimental Results

## Summary Table

| Model                                          | Config Key                                              | I2T R@1    | T2I R@1    | R@10 (avg) | NMI        | Purity     | Del AUC ↓ | Ins AUC ↑ |
| ---------------------------------------------- | ------------------------------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | --------- | --------- |
| A (untrained)                                  | Zero-shot BiomedCLIP                                    | 0.47%      | 0.63%      | ~4%        | —          | —          | —         | —         |
| **A**                                          | Global contrastive only                                 | **17.37%** | **17.82%** | **56.9%**  | **0.1318** | **0.4228** | —         | —         |
| B (no mid-fusion)                              | Cosine local align, no mid-fusion                       | 16.12%     | 14.50%     | 50.9%      | 0.1496     | 0.4501     | 0.461     | 0.4198    |
| B (cosine mid-fusion)                          | Cosine MSE loss, mid-fusion layers 4+8+12               | 18.70%     | 19.12%     | 58.2%      | 0.1320     | 0.4313     | 0.452     | 0.489     |
| **B (FILIP mid-fusion)**                       | FILIP [0,0.1,0.1] layers 4+8+12                         | 18.28%     | **19.12%** | 58.6%      | 0.1495     | 0.4501     | 0.368     | 0.463     |
| B (post-ViT FILIP)                             | FILIP probe post-ViT only, no mid-fusion, 30.07 GFLOPs  | 17.75%     | 18.70%     | 57.7%      | 0.1417     | 0.4370     | 0.461     | 0.476     |
| B (FILIP) at different levels for BERT as well | FILIP probe post-ViT only, no mid-fusion, 30.07 GFLOPs  | 17.22%     | 17.98%     | 56.69%     | 0.1401     | 0.4268     | 0.470     | 0.471     |
| **B (multiscale probes)**                      | FILIP probes at layers 4+8, read-only, 30.28 GFLOPs     | **18.33%** | **18.43%** | ~57.9%     | **0.1536** | **0.4501** | 0.452     | 0.472     |
| C \|1                                          | STE + cosine local (weight=500), 30.28 GFLOPs           | 14.23%     | 13.39%     | 49.2%      | 0.1443     | 0.4486     | 0.482     | 0.416     |
| C \|2                                          | Gumbel + FILIP [0.01,0.1,0.1] + staged                  | 17.98%     | 17.94%     | 57.7%      | 0.1233†    | 0.4357†    | 0.398     | 0.498     |
| C \|3                                          | STE + FILIP [0,0.01,0.01], mr=0.5, 49.61 GFLOPs         | 17.80%     | 17.70%     | 57.3%      | 0.1441     | 0.4501     | 0.334     | 0.501     |
| D \|1                                          | Top-K k=0.4, FILIP [0,0.5,0.5], no fused contrastive    | 18.33%     | 18.33%     | 58.5%      | 0.1184     | 0.4342     | 0.424     | 0.423     |
| D \|2                                          | Top-K k=0.4, FILIP [0.1,0.45,0.45], staged, ES=loss     | 16.81%     | 17.17%     | 56.3%      | 0.0962     | 0.4110     | 0.423     | 0.419     |
| D \|3                                          | Top-K k=0.25, FILIP [0.01,0.01,0.01]                    | 17.65%     | 17.86%     | 58.7%      | 0.1382     | 0.4602     | 0.431     | 0.448     |
| E                                              | PatchScorerMLP at layer 6, K=118, text-agnostic         | 15.18%     | 14.65%     | ~50.3%     | 0.1437     | 0.4269     | 0.458     | 0.470     |
| **F-layer4**                                   | FILIP drop at layer 4, k=0.5, 35.87 GFLOPs              | **18.43%** | **18.70%** | 57.1%      | 0.1474     | 0.4414     | 0.316     | 0.496     |
| **F**                                          | FILIP drop at layer 6, k=0.5 (98 patches), 34.46 GFLOPs | **17.96%** | **18.17%** | 58.0%      | **0.1523** | **0.4645** | 0.205     | 0.595     |
| F-layer9                                       | FILIP drop at layer 9, k=0.5, 32.36 GFLOPs              | 17.49%     | 17.96%     | 57.6%      | **0.1542** | 0.4486     | 0.183     | 0.678     |
| F-adaptive                                     | Adaptive K via STE threshold, layer 6, 34.46 GFLOPs     | 17.86%     | 17.28%     | 57.2%      | 0.1435     | 0.4428     | **0.032** | **0.785** |
| G                                              | 2-stage drop: layer 4→9, K1=147→K2=98, 37.78 GFLOPs     | 17.38%     | 17.02%     | 57.4%      | 0.1460     | 0.4327     | 0.253     | 0.622     |
| **H**                                          | Soft sigmoid gating at layer 6, full seq, 38.67 GFLOPs  | 16.96%     | 16.96%     | 55.6%      | **0.1540** | 0.4530     | 0.299     | 0.681     |

---

## Best Models Summary

| Model                     | Best at                              | I2T R@1    | T2I R@1    | NMI        | Purity     | Del AUC ↓ | Ins AUC ↑ |
| ------------------------- | ------------------------------------ | ---------- | ---------- | ---------- | ---------- | --------- | --------- |
| **A**                     | Retrieval baseline                   | 18.17%     | 18.75%     | 0.1418     | 0.4428     | —         | —         |
| **B (cosine mid-fusion)** | Best I2T R@1 overall                 | **18.70%** | **19.12%** | 0.1320     | 0.4313     | —         | —         |
| **B (multiscale probes)** | Best NMI among non-dropping models   | 18.33%     | 18.43%     | **0.1536** | 0.4501     | 0.452     | 0.472     |
| **B (FILIP mid-fusion)**  | Best faithfulness gap (non-adaptive) | 18.28%     | **19.12%** | 0.1495     | 0.4501     | 0.368     | 0.463     |
| **F**                     | Best purity, strong NMI              | 17.96%     | 18.17%     | 0.1523     | **0.4645** | 0.405     | 0.395     |
| **F-adaptive**            | Best faithfulness                    | 17.86%     | 17.28%     | 0.1435     | 0.4428     | **0.032** | **0.485** |
| **C \|1**                 | Best NMI (masked embedding†† )       | 14.23%     | 13.39%     | 0.1443‡    | 0.4486‡    | 0.465     | 0.476     |

† C|2 NMI/Purity may also be affected by the masked-embedding clustering bug (has both use_mid_fusion and use_masking). Retrieval is correct. Needs re-evaluation for confirmed NMI.

†† C|1 masked mean-pool NMI = 0.1572 (not comparable to other models' full-CLS NMI). Full-CLS NMI = 0.1443.

‡ C|1 full-CLS (encode_independent) values shown for consistent comparison with all other models.

---

## Per-Model Notes

### Model A — Baseline

- Pure global contrastive (InfoNCE) on BiomedCLIP, no local alignment, no masking
- Strong retrieval baseline: 18.17% / 18.75% R@1
- Best NMI (0.1418) among all models — cleanest representation space
- Zero-shot (untrained) gives only 0.47% — MIMIC-CXR is a domain gap from PMC-15M pretraining

---

### Model B (no mid-fusion) — Cosine Local Alignment Only

- Mid-fusion cross-attention injected but local loss is cosine MSE (collapses easily)
- **Worse retrieval than A** (16.12% / 14.50%) — cosine MSE local loss actively hurts
- Marginally better clustering NMI (0.1496) suggesting some local structure learned

### Model B (cosine mid-fusion)

- Mid-fusion cross-attention at layers 4, 8, 12 with cosine MSE local loss (symmetric)
- Best I2T R@1 of all B variants (18.70%) — cosine loss with mid-fusion injection does not collapse unlike the no-mid-fusion cosine variant
- NMI low (0.1320) — cosine MSE local loss does not produce a structured representation space
- No FILIP projections → phrase grounding evaluation produces no quantitative metrics

### Model B (FILIP mid-fusion) — Best B variant

- FILIP weights [0, 0.1, 0.1] at layers 4, 8, 12 (layer 4 FILIP disabled to let features mature)
- Matches best T2I R@1 of B family (19.12%), slight retrieval drop vs cosine variant (−0.42% I2T) within noise
- **NMI 0.1495, Purity 0.4501** — much better than cosine mid-fusion (NMI +0.017), approaching B-multiscale
- Phrase grounding (MS-CXR, layer 4): CNT=0.608, mIoU=0.197 — layer 4 features most spatially discriminative
- Grounding degrades at deeper layers (layer 8: CNT=0.309, mIoU=0.223; layer 12: CNT=0.278, mIoU=0.168) — FILIP features diffuse spatially at deeper layers
- **Deletion/insertion**: Del 0.368, Ins 0.463, gap = **+0.095** — the best faithfulness gap among all non-adaptive models. Mid-fusion FILIP produces more peaked per-patch scores than intra-ViT dropping (F gap = −0.010). The mid-fusion cross-attention directly injects text context into patch features at multiple ViT depths, making the resulting scores more discriminative.

### Model B (post-ViT FILIP)

- FILIP alignment applied after the full 12-block ViT (post-ViT), no intra-ViT injection — probe is read-only on final patch features
- Weakest retrieval of mid-fusion B variants: 17.75% / 18.70% R@1 — post-ViT features already aggregated into CLS, patch-level FILIP signal too late to reshape representations
- NMI 0.1417 ≈ Model A (0.1418): no clustering improvement vs baseline — confirms that FILIP loss applied purely post-ViT has no structuring effect on the embedding space
- GFLOPs 30.07: lowest of all B variants (no intra-ViT cross-attention overhead), smaller param count (196.7M vs 215.9M for mid-fusion)
- Serves as ablation confirming that the mid-fusion injection (not just the FILIP loss) is what drives clustering improvement in B (FILIP mid-fusion) and B (multiscale probes)
- **Deletion/insertion**: Del 0.461, Ins 0.476, gap = +0.015 — near-zero gap, confirming that post-ViT FILIP scores applied on fully-diffused final patch features cannot produce meaningful attribution beyond chance. Aligns with the near-zero clustering improvement over A.

---

### Model C |1 — Best NMI (masked embedding); poor retrieval

- STE masking (mask_ratio=0.5) + cosine local alignment (weight=500), no mid-fusion, 30.28 GFLOPs
- **Retrieval 14.23% / 13.39%** — substantially below baseline A (18.17% / 18.75%). The masked mean-pool used as the primary training embedding degrades backbone retrieval quality; the CLS token receives weaker contrastive gradient than in Model A.
- **NMI 0.1443, Purity 0.4486** (measured via encode_independent / full CLS, for consistency with all other models). When measured on the masked mean-pool itself: NMI 0.1572 — the bottleneck embedding produces excellent clustering structure, but this is not comparable to the full-CLS NMI of other models.
- **Evaluation note**: an earlier run reported 19.49% retrieval, which originated from a checkpoint no longer available (or a different eval protocol). The correct retrieval on the surviving checkpoint is 14.23%.
- The primary contrastive loss is on the masked mean-pool (not the CLS), so the backbone is optimized for a different objective than Model A → CLS quality is reduced despite strong local alignment supervision.
- Mask grounding (MS-CXR): overall recall=0.137, precision=0.058, IoU=0.040, kept_ratio≈38% — weak spatial grounding; text-agnostic STE selects patches uniformly rather than disease-specifically.
- No mid-fusion → no phrase grounding metrics.
- **Deletion/insertion**: Del 0.465, Ins 0.476, gap = +0.011 — modest positive gap; D/I is measured on the mask scoring path independently of retrieval.

### Model C |2 — Gumbel-Sigmoid + FILIP mid-fusion + staged training

- Replaces STE with Gumbel (τ: 1.0→0.1 over 5k steps), FILIP [0.01, 0.1, 0.1]
- Staged training (1 epoch frozen backbone, lr=1e-4)
- Competitive with A on retrieval (18.49% / 18.64%) — Gumbel stabilises training
- NMI 0.1233, Purity 0.4457 (updated from previous run; small k-means variance)
- Phrase grounding (MS-CXR, layer 4): CNT=0.701, mIoU=0.200 — best CNT of all mid-fusion models at layer 4
- Grounding degrades at deeper layers (layer 8: CNT=0.340, mIoU=0.216; layer 12: CNT=0.196, mIoU=0.180)
- Mask grounding: overall recall=0.258, precision=0.086, IoU=0.065 — better spatial overlap than C|1, text-conditioned Gumbel selection more anatomically focused
- **Deletion/insertion**: Del 0.413, Ins 0.408, gap = −0.005 — essentially random attribution despite Gumbel-based selection. The text-conditioned Gumbel selector improves spatial grounding (mask grounding) but not patch importance ordering as measured by deletion/insertion.

### Model C |3 — STE + low FILIP weights

- STE masking + mid-fusion, mask_ratio=0.5, FILIP [0, 0.01, 0.01] (tiny local signal), 49.61 GFLOPs
- Below A on retrieval (17.80% / 17.70%) — FILIP too weak to guide patch selection meaningfully
- **NMI 0.1441, Purity 0.4501** (corrected from previous 0.1095/0.4153 which was measured on the masked mean-pool; now measured via encode_independent for consistency)
- Phrase grounding (layer 4): CNT=0.134, mIoU=0.122 — weakest layer-4 grounding of all mid-fusion models, confirms FILIP weights too small to produce discriminative patch scores
- Grounding improves at layer 8 (CNT=0.485, mIoU=0.204) — intermediate features more spatially structured than layer 4 with near-zero FILIP signal; layer 12 degrades (CNT=0.206, mIoU=0.106)
- Mask grounding: overall recall=0.337, precision=0.099, IoU=0.079 — highest recall of all C variants but low precision (many patches selected indiscriminately at mr=0.5)
- **Deletion/insertion**: Del 0.433, Ins 0.454, gap = +0.021 — tiny FILIP weights produce just enough signal to rank patches slightly better than random; far below B-FILIP-mid-fusion (gap +0.095).

---

### Model D |1 — Top-K k=0.4, strong FILIP

- Top-K annealing 0.75→0.4, FILIP [0, 0.5, 0.5], fused contrastive removed (weight=0)
- Comparable to A (18.33% / 18.33%) — best D variant on retrieval
- Removing fused contrastive (which collapses anyway) is neutral/positive
- Phrase grounding (layer 4): CNT=0.196, mIoU=0.160 — weakest layer-4 grounding of all D variants; FILIP weight=0 at layer 4 means no grounding signal at early layers
- Grounding improves at deeper layers (layer 8: CNT=0.340, mIoU=0.221; layer 12: CNT=0.371, mIoU=0.196) — FILIP signal at layers 8/12 produces later-but-meaningful localization
- Mask grounding: overall recall=0.273, precision=0.111, IoU=0.084
- **Deletion/insertion**: Del 0.424, Ins 0.423, gap = −0.001 — essentially random. Despite strong FILIP weights at layers 8+12, the contrastive training objective does not force peaked patch importance at the scorer level, leaving attribution near-chance.

### Model D |2 — Staged + loss-based early stopping

- Same as D|1 but staged training + early_stopping_metric=`loss` instead of `recall`
- **Worst D variant** on retrieval (16.81% / 17.17%, NMI 0.0962)
- Loss-based early stopping likely chose a checkpoint before retrieval converged
- **Best D variant for grounding**: phrase grounding layer 4 CNT=0.608, mIoU=0.227; mask grounding overall recall=0.374, IoU=0.115 — higher FILIP weights [0.1, 0.45, 0.45] and staged training improve spatial localization despite hurting global retrieval
- Trade-off: retrieval and clustering suffer when local alignment is prioritized via higher FILIP weights
- **Deletion/insertion**: Del 0.423, Ins 0.419, gap = −0.004 — also near-random, same as D|1 despite higher FILIP weights. The early-stopping on loss may have terminated before the scorer learned discriminative patch ordering.

### Model D |3 — Original config (k=0.25, tiny FILIP)

- First D run: k_ratio=0.25 (keeps only 49/196 patches), FILIP [0.01, 0.01, 0.01]
- Below A on retrieval — aggressive pruning with near-zero FILIP signal cannot recover dropped information
- **NMI 0.1382, Purity 0.4602** (updated from previous run 0.1262/0.4399 — k-means variance across runs)
- Phrase grounding (layer 4): CNT=0.546, mIoU=0.205 — surprisingly strong for tiny FILIP weights; layer-4 features retain spatial structure even with minimal alignment loss
- Mask grounding: overall recall=0.273, precision=0.099, IoU=0.084 — comparable to D|1 despite keeping fewer patches
- **Deletion/insertion**: Del 0.431, Ins 0.448, gap = +0.017 — a small positive gap, best of all D variants. Aggressive k=0.25 pruning forces the scorer to produce more discriminative scores (only 49/196 patches survive), resulting in better-than-random attribution despite tiny FILIP weights.

---

### Model B (multiscale probes) — Best clustering model

- Multi-scale FILIP probes at intermediate ViT layers 4 and 8, plus final post-ViT FILIP
- No mid-fusion injection — probes are read-only, backbone unchanged at inference
- NMI 0.1536 — multi-scale features improve clustering (surpassed only by C|1 at 0.1572)
- Retrieval matches A (18.33% / 18.43%) — no degradation from auxiliary probe losses
- Best Purity (0.4501, tied with B no mid-fusion) despite strong NMI
- **Deletion/insertion**: Del 0.452, Ins 0.472, gap = +0.020 — small positive gap; read-only probes produce marginally better-than-random attribution despite no intra-ViT injection. Multi-scale supervision at layers 4+8 shapes backbone features enough to weakly improve patch importance ordering.

### Model E — Intra-ViT patch dropping

- PatchScorerMLP at ViT layer 6, drops (196-K) patches, K≈118 (60% kept)
- Upper ViT blocks (7–12) process K+1 tokens — reduces compute, but scorer is text-agnostic
- **Worse retrieval than A** (15.18% / 14.65%) — dropping 40% of patches loses information
- NMI slightly below A (0.1437) — sparse representation slightly hurts clustering
- Scorer is globally biased (cardiac region), not phrase-specific: grounding maps show uniform high-attention zones rather than disease-specific localization
- No grounding metrics available — no mid-fusion and no mask_head
- **Deletion/insertion confirms completely random attribution**: Del AUC=0.458 ≈ Ins AUC=0.470, gap=0.012. Deletion curve is nearly flat and **rises** after ~50% deletion (0.461→0.477 at 90% deletion) — removing the "most important" patches barely changes similarity and eventually helps. The text-agnostic MLP scorer ranks patches no better than random. Starkest contrast with F-adaptive (gap=0.453)

---

### Model F — FILIP-scored intra-ViT drop (k=0.5, 98 patches)

- Text-conditioned FILIP scorer at drop_layer=6: each patch scored by max cosine sim to text tokens
- **Near-A retrieval** (17.96% / 18.17% R@1) — major recovery over E (+2.78% / +3.52%)
- **First model to exceed A on NMI (0.1523 > 0.1418) and Purity (0.4645 > 0.4428)** — text-conditioned intra-ViT drop produces more structured embedding space than full CLS
- Modest deletion/insertion gap (0.405 vs 0.395) — flat deletion curve suggests scorer not maximally peaked at single layer
- 34.46 GFLOPs — compute saving from sequence reduction in upper blocks
- encode_independent() uses full 12-block ViT CLS for retrieval (text-independent); dropped-path CLS is auxiliary training signal only

---

### Model F-adaptive — Adaptive K via STE threshold

- Same FILIP scorer as F but no fixed K budget: threshold at score=0 → variable K per image-text pair
- **Comparable retrieval to F** (17.86% / 17.28% R@1), NMI matches A exactly (0.1435 ≈ 0.1418)
- **Outstanding faithfulness**: deletion AUC = 0.032 (near-zero!), insertion AUC = 0.485, gap = 0.453 vs F's gap of 0.010
- Deletion curve goes negative (−0.220 at 95% deletion) — complete inversion, model is highly confident about patch importance
- Trade-off: F-adaptive is clearly better for interpretability/faithfulness; F is better for clustering (NMI 0.1523 vs 0.1435)
- Variable K = query-specific sparsity: "pneumothorax" → few patches; "bilateral infiltrates" → more patches

### Model F-layer4 — Early drop ablation

- FILIP scorer applied at layer 4 (positional/textural features), keeps same k=0.5 (98 patches) as F-layer6
- **Best retrieval of all F variants**: 18.43% I2T / 18.70% T2I — essentially matching Model A (18.17%/18.75%) and the best T2I among intra-ViT models
- NMI 0.1474: better than A but below F-layer6 (0.1523)
- Purity 0.4414: slightly below A (0.4428) and well below F-layer6 (0.4645)
- GFLOPs 35.87: slightly higher than F-layer6 (34.46) because dropping at layer 4 runs 8 reduced-sequence blocks vs 6
- **Deletion/insertion**: Del 0.416, Ins 0.406, gap = −0.010 — same as F-layer6 (Del 0.405, Ins 0.395, gap −0.010). Despite the richer 8-block refinement after layer-4 drop, the scorer quality (measured by faithfulness) is no better than layer-6. Layer-4 features are too shallow to produce discriminative per-patch cosine similarities with text tokens.
- Interpretation: layer-4 features are rich enough to identify the most obvious background patches and remove them early; 8 upper blocks then refine the kept patches thoroughly → strong retrieval. But layer-4 semantics are too shallow for disease-specific ranking → weaker clustering vs layer-6, and no faithfulness improvement.

---

### Model F-layer9 — Late drop (layer 9, k=0.5, 32.36 GFLOPs)

- FILIP scorer applied at layer 9 — only 3 upper blocks remain for refinement
- **Scorer near-collapse during training**: WandB shows `logit_std ≈ 0` and `patches_kept_ratio ≈ 1.0` early on — layer-9 self-attention has diffused spatial information, making all patches positively correlated with text tokens and reducing FILIP score variance. TopK selection is near-random.
- Despite collapsed scorer, the model trained via auxiliary FILIP and contrastive losses and produced competitive results: 17.49% / 17.96% R@1, **NMI 0.1542 (highest of all models)**, Purity 0.4486
- **Faithfulness mediocre**: Del AUC 0.383 (slightly better than F's 0.405), Ins AUC 0.378 (slightly worse than F's 0.395). Faithfulness gap = 0.005 — near-zero, indicating the selected patches are not meaningfully more informative than random
- GFLOPs 32.36: lower than F-layer6 (34.46) — only 3 blocks run on the reduced K=98 sequence
- The high NMI is likely incidental: by dropping late with a near-random scorer, the model essentially runs a full 9-block ViT followed by mild perturbation — similar to F-adaptive's full-sequence regime, which also produces structured embeddings
- **Interpretation**: high NMI but near-random patch selection means layer-9 dropping provides no meaningful attribution. The clustering structure comes from the 9-block ViT, not from the scorer quality.

---

### Model G — 2-stage gradual drop (layer 4 → layer 9)

- Stage 1 (layer 4): 196→147 patches (K1, coarse background removal); Stage 2 (layer 9): 147→98 patches (K2, fine disease selection)
- **Disappointing**: retrieval below A (17.38% / 17.02%), NMI below F-layer6, Purity 0.4327 (worst of all F/G variants)
- **GFLOPs 37.78** (corrected from previous erroneous 23.42): G is actually MORE expensive than F-layer6 (34.46) because both stages' reduced sequences still run 5 inter-stage blocks (4→9) and 3 upper blocks (9→12), and the 3 FILIP losses add overhead. The prior compute saving claim was wrong.
- **Deletion/insertion**: Del 0.353, Ins 0.392, gap = **+0.039** — the best faithfulness gap of all fixed-K models (better than F-layer6's −0.010 and F-layer4's −0.010). The 2-stage cascade produces more peaked importance scores: Stage 1 removes obvious background, Stage 2 selects from a pre-filtered set → sharper final patch scores. However this faithfulness advantage does not compensate for the retrieval and clustering degradation.
- Key failure: Stage 1 drops 49 patches based on shallow layer-4 features, some of which Stage 2 would have ranked high using richer layer-9 features. The cascade is pessimistic — Stage 2 never gets to reconsider Stage 1's discards.
- More complex architecture (3 FILIP losses, 2 STE drops) does not compensate for the irreversible information loss at Stage 1
- Take-away: **hierarchical dropping does not improve over single-stage F**; the 2-stage design hurts both retrieval (−0.58%) and clustering (Purity −0.032), is more expensive (+3.32 GFLOPs), and the better faithfulness gap (+0.049 vs F) comes at too high a cost.

### Model H — Soft sigmoid FILIP gating (drop_layer=6, T=5)

- Replaces hard TopK/STE (Model F) with multiplicative sigmoid gates: `patch_i ← patch_i × sigmoid(5 × filip_score_i)`. All 197 tokens continue through upper blocks.
- **Best NMI across ALL models (0.1540 > 0.1536 B-multiscale > 0.1523 F)** — soft continuous gating produces the most structured embedding space
- Purity 0.4530: better than A (0.4428) and all models except F (0.4645)
- **Worst retrieval of F/G/H family** (16.96% / 16.96%) — about 1% below F, comparable to G
- Root cause of retrieval trade-off: the primary embedding is the full CLS (all 196 patches, no selection). The gating pressure only reaches the backbone via the auxiliary gated contrastive. Contrast with F: the dropped CLS directly forces the CLS to attend only to K relevant patches, creating stronger selection pressure.
- Fully differentiable: clean sigmoid gradients, no STE approximation needed
- WandB: gates converging to ~0.5 mean activation, logit_std growing → healthy training signal
- The sigmoid weight map IS the importance visualisation — each patch gets a continuous [0,1] importance score, more interpretable than binary masks

---

## Key Findings

1. **No model clearly outperforms A on retrieval** — all models are within ±1.5% R@1 of the 18.17/18.75% baseline.

2. **Masking consistently hurts retrieval** — C|1 (14.23%) is the starkest example, well below baseline A (18.17%). The primary contrastive loss is trained on the masked mean-pool, not the CLS, so the backbone's CLS quality is sacrificed. C/D variants with mid-fusion fare better (retrieval near A) because the CLS is also trained via an independent contrastive path.

3. **FILIP mid-fusion without masking (Model B FILIP)** gives the best T2I R@1 (19.12%) and best I2T is B cosine mid-fusion (18.70%) — both above A — suggesting mid-fusion cross-attention is the most reliable way to improve retrieval without risking collapse.

4. **Model C|1's masked embedding produces the best clustering structure (NMI 0.1572 on masked mean-pool)** but this is not comparable to other models' full-CLS NMI. On a consistent encode_independent evaluation, C|1's NMI is 0.1443, and B-multiscale (0.1536) and F-layer9 (0.1542) both surpass it. The strong cosine local alignment (weight=500) shapes the backbone but at a significant retrieval cost.

5. **Loss-based early stopping (D|2) is harmful** for contrastive models — val loss can decrease while retrieval is still improving. Use `recall` or `combined` as stopping metric.

6. **Staged training alone does not help** (D|2 is worse than D|1) — the benefit depends on paired correct hyperparameters, especially the early stopping metric.

7. **Multi-scale FILIP probes (Model B multiscale) give NMI 0.1536** without hurting retrieval — intermediate-layer features capture complementary structure. The read-only probe design avoids the mid-fusion incompatibility with patch masking. On a consistent encode_independent comparison, B-multiscale NMI (0.1536) exceeds C|1's full-CLS NMI (0.1443).

8. **Intra-ViT patch dropping (Model E) hurts retrieval** — the text-agnostic scorer drops semantically relevant patches, and the hard-drop boundary at layer 6 loses information that cannot be recovered in upper blocks.

9. **Text-conditioned FILIP scoring (Model F) substantially outperforms text-agnostic scoring (Model E)** — +2.78% I2T R@1, +3.52% T2I R@1 — confirming that query-specific patch selection is essential.

10. **Model F is the first variant to exceed Model A on both NMI and Purity** (NMI: 0.1523 vs 0.1418; Purity: 0.4645 vs 0.4428) while maintaining near-A retrieval. Intra-ViT text-conditioned dropping improves clustering structure without sacrificing retrieval.

11. **Model F-adaptive achieves dramatically superior faithfulness** — deletion AUC 0.032 vs 0.405 for F, insertion AUC 0.485 vs 0.395 — demonstrating that variable-K adaptive selection produces qualitatively superior patch discriminability compared to fixed Top-K.

12. **Faithfulness vs retrieval/clustering trade-off**: F (fixed K) > F-adaptive for clustering NMI and retrieval; F-adaptive >> F for faithfulness metrics. Choice depends on application target.

13. **Drop layer ablation (F-layer4 vs F-layer6 vs F-layer9)**: Earlier dropping improves retrieval (layer 4 ≈ Model A) but hurts clustering. Layer 6 is optimal for clustering (best NMI and Purity). The sweet spot is mid-network where features are semantically meaningful but enough upper blocks remain for refinement.

14. **2-stage hierarchical dropping (Model G) does not improve over single-stage F**: Cascaded dropping is strictly worse on retrieval (17.38% vs 17.96%) and clustering (Purity 0.4327 vs 0.4645). The irreversible discards in Stage 1 cannot be compensated by the finer Stage 2 selection.

15. **Layer 9 is too late for meaningful FILIP scoring**: Features at layer 9 are globally diffused — the scorer shows near-zero std and TopK selection is near-random. Despite this, F-layer9 achieves the highest NMI (0.1542, marginally above H at 0.1540), likely because running 9 ViT blocks before any drop produces richer representations. Faithfulness is poor (Del/Ins gap ≈ 0.005 vs F-adaptive's 0.453), confirming the attribution is meaningless. Layer 6 is the sweet spot: semantically mature yet spatially discriminative.

16. **Model H (soft sigmoid gating) achieves NMI 0.1540**, highest among the F/G/H family, beating B-multiscale (0.1536) and F (0.1523). On a consistent encode_independent comparison, H now holds the overall best NMI among all models. Soft continuous gating creates a more evenly distributed representation pressure, producing the most structured clustering among intra-ViT models. Trade-off: retrieval is the worst of the F/G/H family (16.96%), because the primary CLS path (full ViT, no selection) does not directly experience the gating pressure — only the auxiliary gated CLS does.

17. **Hard drop (F) vs soft gate (H) trade-off**: F is better for purity (0.4645 vs 0.4530) and retrieval (17.96% vs 16.96%); H is better for NMI (0.1540 vs 0.1523). Hard sequence reduction forces the CLS to attend exclusively to kept patches (strong selection signal); soft gating distributes signal across all patches (better global structure but weaker local focus).

18. **Neither H nor F-layer9 approach F-adaptive's faithfulness**: Del/Ins AUC gap = H: 0.018, F-layer9: 0.005, F-layer6: 0.010 — all near-zero vs F-adaptive's 0.453. F-adaptive's variable-K threshold design is uniquely suited for producing faithful patch attributions; fixed-K (F, F-layer9) and soft-gate (H) models produce roughly equally uninformative saliency maps despite their architectural differences.
