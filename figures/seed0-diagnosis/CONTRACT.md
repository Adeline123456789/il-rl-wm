# Seed0 diagnostic figure contract

Question: did U0 retain the behavior achieved by its C parent, and what value changes accompanied its failure?
Claim: in this single development seed, U0 lost success late while matched C continuation remained successful; separate fixed-input task values also shifted downward. This is association, not causal attribution.

Quantitative grid, Python/matplotlib only, 183 × 85 mm. No journal submission claim. Editable PDF/SVG and 600 dpi PNG, ≥6 pt labels. Source CSVs and input hashes accompany the figure.

- a: primary behavioral comparison. All five registered validation checkpoints, 16 whole episodes per method, same C parent and equal 262144 additional online interactions. Four U0 methods plus C continuation; inherited interactions included on x axis.
- b: numerical training diagnosis. All recorded Q1 minibatch means for U0 and C continuation. Batches differ between runs; not fixed-input calibration, and soft Q is not bounded by pure-task return 100.
- c: orthogonal fixed-input check. Every one of 2048 expert and 2048 online sampled inputs, with replacement, before/after snapshots. Box median/IQR, whiskers5–95%, all outliers shown. Variation is across states, not across training seeds.

All panels use one training seed, not independent replication. No significance test, confidence interval or training-seed error band. Validation seed10000; no final test. Figure omits other experiments because it diagnoses this matched continuation comparison; A/B initial-stage results remain in raw artifacts and the batch report. No checkpoint selection: the intermediate snapshot is used only for diagnosis, and the failed final checkpoint remains final.

Before export: measured panel alignment at1.5pt. After export: source, PDF glyph and collision checks, then visual inspection of each panel and the whole figure. Missing checks remain explicitly pending.
