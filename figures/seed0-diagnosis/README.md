# PickCube seed0 failure diagnosis

**Single development training seed; descriptive evidence, not a multi-seed conclusion.**

(a) U0 and C continuation share a C parent trained for262144 interactions, then receive262144 additional interactions each. Every point is success-once among16 full validation episodes (seed10000). All four U0 evaluation paths and the C control are shown; line segments only connect measured checkpoints. The failed final checkpoint remains the final result.

(b) All recorded soft Q1 training-minibatch means. U0 and C batches differ. The soft objective contains entropy; values exceeding100 alone do not prove pure-task value overestimation.

(c) Pure-task values of deterministic student mean actions on the same2048 expert and2048 online inputs, sampled with replacement. Before is U0 segment step196608 (458752 total interactions), after is fixed final (524288 total). Boxes show median and IQR, whiskers5th–95th percentiles, with every outlier shown. This is variability across fixed inputs, not training-seed uncertainty. Model behavior and values change together; these plots do not establish the causal direction.

No final-test data, hypothesis test, p-value or cross-seed confidence interval. Raw paths and SHA256 hashes are in sources.json; validation.csv, critic.csv and fixed-input-values.csv contain every plotted numerical observation/aggregate. Original episode/sample arrays remain in artifacts.

Re-render with the locked main Python environment and the nature-figure QA scripts:

```sh
.venv/bin/python scripts/plot_seed0_diagnosis.py --qa-scripts <nature-figure-scripts-directory>
```

The source-data extraction and figure construction are also available as `build_figure(output)` in that script. The QA helper is an external rendering check, not a main training dependency.

Final rendered QA: panel alignment PASS (1.5pt tolerance, no exemptions), editable PDF text minimum6pt, collision audit0fail/0warn. Source audit19pass/2warn/0fail. PNG-only raster warning is accepted because this is a repository report with PDF/SVG and600dpi preview, not a TIFF journal submission. Uncertainty warning is accepted because there is only one training seed; no inferential band is invented.

Visual inspection: (a) all five curves and marker styles are legible, including final collapse; overlapping SAC/no-IL values are retained. (b) both control/U0 curves are visible on a full linear axis; the spike is not cropped. (c) four fixed-input distributions, whiskers and outliers are visible. Labels, legend and plot areas have no overlap. All three panels use the source data without smoothing or trimming. Earlier failed QA logs are preserved; the final outputs use the corrected panel-label transform.
