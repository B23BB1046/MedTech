# **Report on Model Performance and Experimental Findings**

## **1\. Objective**

The goal of these experiments was to evaluate whether the available questionnaire-derived feature sets could support reliable prediction of two classification targets, **T** and **N**, and whether the **pain** outcome was better modeled as a regression task. The experiments were designed to compare multiple model families, test ordinal formulations, test latent-score-to-threshold approaches, and examine whether PCA could improve performance relative to non-PCA baselines. The analysis also tested whether class imbalance handling, especially for the rare T class `0`, could materially improve predictability.

## **2\. Experimental Setup**

The work was organized around two feature regimes:

* **`vpk`** for **T classification**  
* **`all`** for **N classification**

The classification experiments included three approaches:

1. **Standard multiclass classification**  
2. **Ordinal classification**  
3. **Latent regression with two decoding strategies**  
   * direct rounding  
   * thresholding from predicted latent scores

For class imbalance, the T target was further modified by duplicating the rare class `0` to ensure it had at least 10 samples in training folds. All performance comparisons were made using stratified cross-validation so the class distributions were preserved in evaluation. The PCA experiments were run separately after the non-PCA benchmarks, using several component settings to test whether dimensionality reduction improved generalization.

## **3\. Summary of the Main Findings**

The results show a clear structural split between the targets.

### **T classification**

T on `vpk` is **not reliably learnable** with the current features and methods. Even after oversampling class `0`, testing ordinal models, and applying latent-score decoding, the best models still failed to clearly beat the strongest dummy baseline in balanced accuracy. The best observed balanced accuracies for T remained in the roughly **0.24–0.31** range, which is too weak to support robust deployment.

### **N classification**

N on `all` is **weakly learnable**, but only marginally above baseline. The best models achieved balanced accuracy around **0.31**, compared with dummy baselines around **0.25–0.27**. That indicates there is some signal, but it is modest and unstable. The results do not support the conclusion that N is strongly predictable from the current features.

### **PCA**

PCA did **not produce a major breakthrough** for either target. For T, PCA did not change the overall conclusion that the signal is weak. For N, PCA occasionally improved balanced accuracy slightly, especially in latent or SVM-style models, but the gains were small and often offset by lower macro-F1. PCA therefore appears to be a mild stabilization tool, not a remedy for weak label-feature correlation.

## **4\. Detailed Analysis by Target**

### **4.1 T classification on `vpk`**

The most important finding is that the T target does not behave like a stable, learnable 4-class problem under the current representation. Standard models such as SVM, RandomForest, ExtraTrees, and ElasticNet logistic regression produced only modest scores. Their balanced accuracies clustered near the dummy stratified baseline, and in many cases remained below it. This is a strong sign that the models are not extracting a robust class boundary, even when class imbalance is partially addressed.

The ordinal approach did not solve this. Ordinal SVM achieved high raw accuracy, but that was misleading because balanced accuracy and macro-F1 remained low. This means the ordinal model was still collapsing toward majority or adjacent classes rather than learning a meaningful ordered structure. The same pattern held for the ordinal tree-based models.

The latent-score approaches were also unconvincing. Regression-to-score followed by thresholding performed slightly better than some standard models in isolated cases, but still did not exceed the better baseline in a meaningful way. This suggests that T may not be well represented by a simple monotonic latent variable in the current features. In practical terms, T is either too noisy, too sparse, or too weakly encoded in `vpk` for reliable prediction.

The class-0 duplication helped only in the narrow sense of making training less degenerate. It did not create new information, and it did not materially improve predictive separation. That is expected: oversampling can stabilize training, but it cannot generate missing structure. The rare class remains a major limitation.

### **4.2 N classification on `all`**

N is more promising than T, but only modestly so. Standard models on `all` produced the best non-PCA results, with RandomForest and ExtraTrees reaching balanced accuracy around **0.287**, which is only a small improvement over the dummy stratified baseline. This shows that N carries some signal in the feature space, but the signal is weak and far from decisive.

Ordinal classification did not substantially improve the situation. The ordinal tree models and ordinal logistic model were competitive but not clearly superior to the standard approaches. This suggests that N is not strongly ordinal in the modeling sense, or at least that the ordering is too weak to be exploited consistently by the available models.

The strongest N result came from the latent-threshold SVR model, which reached balanced accuracy around **0.312** and macro-F1 around **0.307**. That is the best evidence in the entire set that N contains a weak but real latent structure. However, the improvement over baseline is still small. The correct interpretation is not that the problem has been solved, but that N is marginally learnable and may benefit from cleaner features, stronger preprocessing, or more informative labels.

## **5\. PCA Findings**

PCA was tested on both `vpk` and `all` across multiple component sizes. The outcome was consistent: PCA did not create a clear step change in performance.

For T on `vpk`, PCA did not overcome the baseline barrier. Some component settings produced small numerical fluctuations, but no result changed the overall conclusion that the target is weakly modeled by the current features. The PCA-transformed T experiments remained constrained by the same imbalance and weak separability problems observed in the raw feature space.

For N on `all`, PCA produced some small gains in balanced accuracy for latent-score models and some tree-based models. The best PCA results were slightly above the non-PCA versions in a few cases. However, those gains were modest and often came with lower macro-F1 or instability across component choices. The overall result is that PCA can simplify the representation, but it does not reveal a hidden strong signal.

A practical side note is that some PCA runs produced convergence warnings for the SAGA-based logistic models. That means those specific results should be treated as approximate rather than definitive. The broader trend is still reliable, but the exact values for those runs are less trustworthy.

## **6\. What the Results Imply**

The experiments point to three important conclusions.

First, **pain-like or continuous outcomes are more naturally modeled than the classification targets in this dataset**, but that finding was already established earlier and is not the focus of the present report. The classification tasks are harder because they appear to be derived from weakly separated or noisy latent structure.

Second, **T and N do not have the same relationship to the feature sets**. T appears to be less recoverable from the current representation than N. N has at least a small predictive signal in `all`, while T on `vpk` remains very weak even after imbalance corrections. This is a strong indicator that the two targets should not be treated identically in future modeling.

Third, **the current bottleneck is not model complexity**. More complex formulations—ordinal heads, latent-score decoding, PCA, and class oversampling—did not substantially change the outcome. The dominant limitation appears to be the information content and separability of the features relative to the labels.

## **7\. Factors That Could Improve Performance**

The results suggest several plausible avenues for improvement:

1. **Label refinement**  
   T class `0` is too sparse. If clinically acceptable, it may need to be merged, removed, or redefined.  
2. **Feature engineering rather than feature compression**  
   PCA was not transformative. More targeted feature selection or domain-driven feature construction is likely to be more useful.  
3. **Alternative label formulations**  
   T and possibly N may be better treated as ordinal or latent-continuous problems, but only if the underlying label semantics justify that formulation more strongly than the current data indicate.  
4. **More data**  
   This is especially important for the rare T classes and for N if the objective is to push beyond marginal gains.  
5. **Separate task-specific pipelines**  
   The evidence favors separate modeling strategies for T and N rather than a shared architecture at this stage.

## **8\. Final Conclusion**

The experiments support a clear conclusion: **T on `vpk` is currently not reliably predictable, and N on `all` is only weakly predictable**. Standard classification, ordinal classification, latent-score regression, and PCA all produce consistent evidence of limited separability. The best N models are only modestly above baseline, while T remains at or below the stronger dummy baseline in balanced accuracy. The practical implication is that future work should prioritize label design, feature refinement, and possibly class consolidation over larger or more complex models.

If you want, I can turn this into a clean **report-style prose version with section numbering and polished academic tone** suitable for direct inclusion in a document.

