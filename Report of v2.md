# **Report on the v2 Experimental Outputs**

## **1\. Scope of the v2 notebook**

The v2 notebook evaluated the dataset in **nine configurations**: `value_small`, `value_big`, `value_ultra`, `vpk_small`, `vpk_big`, `vpk_ultra`, `all_small`, `all_big`, and `all_ultra`. Each configuration used the same base dataset of **226 samples** and the same class distributions, with T distributed as `{1: 35, 2: 133, 3: 53, 0: 5}` and N distributed as `{2: 48, 1: 106, 0: 58, 3: 14}`. The training set after balancing was **182 samples** and the test set was **46 samples** in every run. The notebook then reported three outputs for each configuration: **T classification**, **N classification**, and **pain regression**.

The model family was tested at three capacity levels: **small**, **big**, and **ultra**. The repeated `GradScaler` / `autocast` warnings show that the notebook used PyTorch mixed-precision training, so the v2 system was clearly a neural-network-based setup rather than a classical ML baseline. The important point is that the same dataset splits and target structure were reused across all capacities, so differences in output are attributable mainly to feature regime and model size rather than changing data.

## **2\. High-level result pattern**

The overall result is very consistent: **pain regression was learnable, while T and N classification remained substantially weaker**. The best regression scores were notably stronger than the classification scores across all nine configurations, especially for the `all` feature family. In contrast, classification performance was limited by imbalance, class overlap, and heavy collapse toward the majority/intermediate classes.

A second major pattern is that **increasing model size did not consistently improve performance**. In many cases the larger models produced **higher confidence** and **lower entropy**, but not better balanced accuracy or macro-F1. That is an important sign of overconfident fitting rather than genuinely improved separability.

## **3\. Data structure and imbalance**

The class distributions explain a large part of the outcome. For T, the dataset is dominated by class `2` with **133 samples**, while class `0` has only **5 samples**. For N, the largest class is `1` with **106 samples**, while class `3` has only **14 samples**. This is a severe imbalance regime, especially for T, where the rarest class is nearly absent. Any model trained in this setting will tend to optimize the majority classes unless the architecture and loss are strongly rebalanced.

This imbalance also means that raw accuracy is misleading. Several models show moderately good accuracy while balanced accuracy remains poor. That tells us the models are learning a frequency-biased decision rule rather than a truly discriminative one. For this dataset, **balanced accuracy and macro-F1 are the right primary metrics**.

## **4\. Detailed analysis by feature family**

### **4.1 `value` feature family**

The `value` configurations are the strongest family for **pain regression**. The best pain result in this family is `value_ultra`, with **R² \= 0.5466**, **MAE \= 0.1101**, and **RMSE \= 0.1450**. `value_small` also performs reasonably well with **R² \= 0.4718**, while `value_big` is slightly worse at **R² \= 0.4080**. That is a clear sign that the `value` features carry meaningful signal for the continuous pain target.

For **T classification**, the `value` family is weak. Balanced accuracy is only **0.2449** for `value_small`, **0.2386** for `value_big`, and **0.2706** for `value_ultra`. Even where raw accuracy is not terrible, the confusion matrices show that the model mostly predicts class `2` and struggles badly with the minority classes, especially class `0`.

For **N classification**, the `value` family is also limited. Balanced accuracy ranges from **0.2489** to **0.2829**, which is not materially strong. So `value` features are most useful for pain, but only marginally helpful for T and N. The outputs suggest that these features are not the main driver of categorical structure.

### **4.2 `vpk` feature family**

The `vpk` family is the most informative family for the classification tasks, especially **N**. The strongest result in the entire v2 output for N is `vpk_ultra`, with **balanced accuracy \= 0.5108** and **macro-F1 \= 0.4960**. That is the only configuration that moves N into a genuinely competitive region. This is strong evidence that N has real signal in the prakriti-style feature block.

For **T**, `vpk_small` is the best-balanced model in this family with **balanced accuracy \= 0.3535** and **macro-F1 \= 0.3208**. `vpk_big` is weaker at **0.2761**, while `vpk_ultra` recovers to **0.3345**. The pattern is not monotonic, but it is still better than the `value` family for T. That means T is more aligned with the `vpk` feature structure than with the `value` structure.

For **pain regression**, `vpk` performs poorly. `vpk_small` gives **R² \= \-0.1134**, `vpk_big` is approximately zero at **\-0.0037**, and `vpk_ultra` remains negative at **\-0.1171**. This is a very strong signal that `vpk` does not encode the continuous pain target well. In practical terms, `vpk` is a classification-oriented representation, not a pain-regression representation.

### **4.3 `all` feature family**

The `all` family is the best overall for **pain regression** and one of the best compromise families for classification. The strongest pain result is `all_big`, with **R² \= 0.5574**, **MAE \= 0.1160**, and **RMSE \= 0.1432**. `all_ultra` is still strong at **R² \= 0.4287**, while `all_small` is weaker at **R² \= 0.3614**. This shows that the combined feature space is useful for pain and that the larger model capacity helped here more than in many other settings.

For **T classification**, `all_ultra` gives the best balanced accuracy in this family at **0.3303**, followed by `all_big` at **0.3123** and `all_small` at **0.2854**. So the combined feature block does help T somewhat, but not enough to solve the class-separation problem. The model still appears to collapse toward central classes.

For **N classification**, `all_big` is the best of the three with **balanced accuracy \= 0.3088** and **macro-F1 \= 0.3109**. `all_small` is weaker, and `all_ultra` actually drops again. That non-monotonicity is important: more capacity does not reliably improve categorical prediction.

## **5\. Model-capacity behavior: small vs big vs ultra**

The capacity trend is not stable. In several cases, going from small to big or ultra increases confidence substantially, but does not yield a proportional performance gain. For example, `all_ultra` has very high confidence and low entropy, yet its N performance is worse than `all_big`. Likewise, `vpk_ultra` is extremely confident, but the actual T gain over `vpk_small` is modest. That is a classic symptom of **overconfident but weakly calibrated decision boundaries**.

This is why the confidence and entropy outputs matter. The better-performing models do not always have the highest confidence. In fact, the most confident models are sometimes the ones that are most likely to be overfitting the majority structure. The outputs therefore support a cautious interpretation of “bigger model equals better model.” In v2, that was not true in a consistent or reliable way.

## **6\. Confusion-matrix interpretation**

The confusion matrices show a recurring pattern across the whole notebook: predictions concentrate in the middle classes. For T, class `2` is the dominant prediction almost everywhere. For N, predictions cluster around classes `1` and `2`. That means the model is not learning the edges of the label space well, especially the rare class `0` and the higher-end classes in some settings.

This is especially visible for T. Even when accuracy looks acceptable, the matrix reveals that the model is rarely making genuinely correct minority-class predictions. So the real problem is not just class imbalance in the abstract; it is the combination of imbalance plus weak class separability and likely ordinal overlap between adjacent classes.

## **7\. Main inferences from the v2 results**

The most defensible inference is that **pain is a much more learnable target than T or N in this version of the dataset**, and it is learned best from the `all` feature family. The second inference is that **T is more aligned with `vpk` than with `value`**, but even there the performance is only moderate. The third inference is that **N has some real predictive signal in the combined feature space and especially in `vpk_ultra`, but the signal is still not strong enough for robust deployment without further feature or label refinement**.

A further, very important inference is that **model complexity was not the limiting factor**. The jump from small to big to ultra did not systematically improve results. That means the bottleneck is likely the data geometry: class imbalance, label overlap, limited sample size, and perhaps weak semantics between features and targets. The architecture can only exploit what the data already contain.

## **8\. Final conclusion**

The v2 notebook shows a coherent but restrictive picture. `value` is best for pain, `vpk` is best for classification, and `all` is the best compromise when one wants a single representation. However, none of the classification settings produce strong balanced accuracy, and the rare T class remains a major obstacle. The results support the view that the current modeling problem is constrained more by **signal quality and class structure** than by model capacity. Future improvement is more likely to come from better target design, better imbalance handling, and possibly feature selection than from simply enlarging the model further.

If you want, I can turn this into a cleaner **formal report format** with an executive summary, methodology, findings, limitations, and recommendations section.

