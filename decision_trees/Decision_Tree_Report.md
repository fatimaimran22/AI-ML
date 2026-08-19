# Decision Tree Classifier — Full Pipeline Report
### Dataset: Pima Indians Diabetes | Beginner-friendly walkthrough

---

## 0. Why this dataset?

I picked the **Pima Indians Diabetes dataset** instead of Titanic, Heart Disease, or Mushroom, for reasons that matter specifically because you're a beginner:

- **All 8 features are numbers (continuous)** — no categorical encoding needed. One less thing to learn before you can focus on the tree itself.
- It has a **real, meaningful class imbalance** (~65% no-diabetes / 35% diabetes), so the class-imbalance questions in this assignment actually mean something on this data.
- It's **small enough to visualize** but not toy-sized (768 patients, 8 features).
- Two of its features (Glucose, BMI) are intuitive enough for a human to look at a 2D decision boundary plot and understand it immediately.
- Mushroom is "too easy" (nearly perfectly separable by one feature), which makes overfitting/pruning boring. Pima is exactly hard enough that pruning *visibly* helps.

**Target column:** `Outcome` (1 = has diabetes, 0 = does not)
**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

---

## 1. Load and Explore

768 rows, 8 features, 1 binary target. No `NaN` values technically exist in this dataset — but that's a trap. Several columns use the number **0** to mean "not measured," even though a real blood pressure or BMI of 0 is medically impossible. Here's how many disguised missing values each column has:

| Column | Zeros (= missing) | Out of |
|---|---|---|
| Glucose | 5 | 768 |
| BloodPressure | 35 | 768 |
| SkinThickness | 227 | 768 |
| Insulin | 374 | 768 |
| BMI | 11 | 768 |

**Beginner note:** For this assignment we deliberately leave these zeros as-is (real cleanup, like imputing them with the column median, is its own lesson). Decision trees are fairly tolerant of this because they only ever ask "is this value above or below a threshold?" — a cluster of disguised zeros just becomes its own little region the tree can carve off with a split, which is exactly what happens to Insulin and SkinThickness below (they end up unused).

**Class balance:** 500 patients with no diabetes (65.1%), 268 with diabetes (34.9%).

![Class Distribution](plots/01_class_distribution.png)

A classifier that *always* predicts "no diabetes" would already be right 65.1% of the time. That number is our floor — any model we build needs to clear it by a meaningful margin to be worth anything.

---

## 2. Split Criterion

We use **Gini impurity** (scikit-learn's default) rather than entropy/information gain. In one sentence: for a binary target like this, Gini and entropy pick almost identical splits in practice, and Gini is cheaper to compute because it avoids the `log()` calculation entropy requires at every single candidate split.

---

## 3. Baseline: Fully-Grown (Deliberately Overfit) Tree

Trained with `max_depth=None`, `min_samples_leaf=1` — i.e., "keep splitting until every leaf is pure or has one sample." This is the classic overfitting recipe.

| Metric | Value |
|---|---|
| Train accuracy | **1.000** |
| Test accuracy | **0.727** |
| Gap | **0.273** |
| Tree depth | 16 |
| Number of leaves | 115 |

That 27-point gap between train and test accuracy is the entire point of this step: the tree has **memorized** the training set (115 leaves for 614 training rows — it's basically drawn a tiny box around clusters of individual patients) rather than learning general rules. It gets *every single* training patient right and is noticeably worse on new data.

---

## 4. Tuning via Cross-Validation

We ran `GridSearchCV` (5-fold stratified) over `max_depth`, `min_samples_split`, and `min_samples_leaf`.

**Best parameters found:** `max_depth=3`, `min_samples_leaf=20`, `min_samples_split=2`
**Best cross-validated accuracy:** 0.759

To visualize *why* depth matters, here's train vs. test accuracy at every depth from 1 to 20:

![Train vs Test vs Depth](plots/03_train_test_vs_depth.png)

Test accuracy peaks around **depth 4** (~0.80) and then the two curves diverge — train keeps climbing toward 1.0 while test flattens and eventually drifts down. That gap opening up *is* overfitting.

---

## 5. Cost-Complexity Pruning (ccp_alpha)

Instead of just capping depth, cost-complexity pruning grows the full tree and then works backward, snipping off the "weakest" branches (the ones that reduce impurity the least relative to how much complexity they add) as `ccp_alpha` increases.

![Pruning Path](plots/04_pruning_path.png)

- **Best alpha found:** 0.00578
- **Test accuracy at that alpha:** 0.799
- **Leaves at that alpha:** 8 (down from 115 in the unpruned tree)

Going from 115 leaves to 8 while *improving* test accuracy (0.727 → 0.799) tells you plainly that the vast majority of those 115 leaves were memorizing noise, not capturing real signal. Only 8 splits' worth of information in this dataset actually generalizes.

This final **pruned tree** (depth 5, 8 leaves) is what we evaluate and report on from here forward.

![Tree Visualization](plots/02_tree_visualization.png)

Reading the root node: the single most useful yes/no question you can ask a Pima patient to predict diabetes is **"is your glucose level above 154.5?"** If yes, 83.8% of patients in that group have diabetes. If no, the tree keeps drilling down through BMI, then age, then glucose again at a lower threshold.

---

## 6. Final Evaluation (Pruned Tree)

We report five metrics, not just accuracy, because the dataset is imbalanced (accuracy alone can be misleading — see Q1 below).

| Metric | Value |
|---|---|
| Accuracy | 0.799 |
| Precision | 0.709 |
| Recall | 0.722 |
| F1 | 0.716 |
| Log loss | 0.510 |
| ROC AUC | 0.804 |

![Confusion Matrix](plots/05_confusion_matrix.png)

Confusion matrix breakdown on the 154 test patients: **84 true negatives, 16 false positives, 15 false negatives, 39 true positives.**

![ROC Curve](plots/06_roc_curve.png)

![Feature Importance](plots/07_feature_importance.png)

The tree relies almost entirely on **Glucose (66%)**, **BMI (18%)**, and **Age (12%)**. Four features — Insulin, DiabetesPedigreeFunction, SkinThickness, Pregnancies — got **0% importance**, meaning the tree never found a split on them that beat the alternatives at any node. This lines up with clinical intuition: glucose level is literally how diabetes is diagnosed.

![Decision Boundary](plots/08_decision_boundary.png)

This is a 2D tree fit on just Glucose and BMI, to make the "blocky rectangle" nature of tree boundaries visible. Notice it's all straight horizontal/vertical lines — never a diagonal or curve, because each split only looks at one feature at a time.

---

## 7. Tracing One Misclassified Example by Hand

Chosen test patient (index 0 among misclassified rows):

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
|---|---|---|---|---|---|---|---|
| 7 | 159 | 64 | 0 | 0 | 27.4 | 0.294 | 40 |

**True label:** 0 (no diabetes) — **Predicted label:** 1 (diabetes)

**Path through the tree:**

1. **Node 0 (root):** Is Glucose (159) ≤ 154.5? **No** (159 > 154.5) → go **RIGHT**.
2. **Leaf node 14:** This leaf contains 99 training patients: 16 with no diabetes, 83 with diabetes. Majority vote = **class 1 (diabetes)**, with 83.8% confidence.

**Why the leaf was wrong for this patient:** the tree reached its verdict after asking exactly *one* question — glucose above 154.5 — and stopped there, because that split alone was "good enough" for the pruned tree's complexity budget. But this specific patient's glucose (159) is only just over the 154.5 threshold, and their BMI (27.4) is fairly low (a genuinely diabetic profile at this glucose level would usually also show a higher BMI). The pruned tree, at only 8 leaves, isn't complex enough to draw a finer boundary that would have separated this patient from the "typical" high-glucose diabetic. This is the accuracy/simplicity trade-off we made in the pruning step, made concrete: we accepted a small number of errors like this one in exchange for a tree that doesn't memorize noise.

---

## 8. Conceptual Questions — Answered

### On the class distribution bar chart (Plot 1)

**Q1. If one class outnumbers the other 9:1, why can a tree that always predicts the majority class still score 90% accuracy — and how does class imbalance bias the split criterion toward ignoring the minority class?**
Accuracy just counts "how many did I get right," so if 90% of rows are the majority class, guessing it every single time is right 90% of the time by definition — the model doesn't need to learn anything. As for the split criterion: Gini/entropy measure *impurity reduction*, and a node dominated by the majority class already has low impurity. A split that only barely improves purity by isolating a handful of minority-class points often reduces impurity less than a split that carves out a larger, purer majority-class chunk — so the greedy algorithm, which always takes the split with the biggest impurity drop, naturally gravitates toward splits that serve the majority class first.

**Q2. Would `class_weight='balanced'` change which splits get chosen, or only what class each leaf votes for?**
It changes **which splits get chosen**, not just the final vote. `class_weight='balanced'` up-weights minority-class samples *before* computing Gini/entropy at every candidate split, so the impurity formula itself changes: a node that's "70% majority / 30% minority" in raw counts might become closer to "50/50" once minority samples count for more. That shifts which split reduces (weighted) impurity the most, changing the actual tree structure, not just relabeling leaves afterward.

### On the visualized decision tree (Plot 2)

**Q3. Why did the algorithm pick Glucose ≤ 154.5 at the root over all others?**
At the root, the algorithm tries every feature and every possible threshold on that feature, computes the Gini impurity reduction each split would produce, and keeps the single split that maximizes that reduction. Glucose at 154.5 produced the single largest drop in Gini impurity across every feature/threshold combination tried — which makes sense, since glucose level is clinically the most direct signal of diabetes.

**Q4. Follow one leaf down from the root — at what depth did it stop, and is it pure, near-pure, or mixed?**
The leaf we traced in Section 7 (node 14) stopped at **depth 1** (right after the root split) and is **near-pure but not pure** — 83 of 99 samples (83.8%) are class 1. A mixed leaf like this means the model isn't fully confident about points landing there: roughly 1 in 6 patients who reach this leaf actually don't have diabetes, so predictions here carry real uncertainty even though the majority vote is "diabetes."

**Q5. If a new point sits exactly at a threshold, which branch does it take?**
scikit-learn's convention is `<=` goes left, so a point exactly equal to the threshold takes the **left branch**. (In our root node, a glucose value of exactly 154.5 would go left, toward "no diabetes" side of the split.)

### On the train-vs-test accuracy vs. depth curve (Plot 3)

**Q6. At what depth does test accuracy peak and then decline while train keeps climbing?**
Test accuracy peaks at **depth 4** (~0.80) then flattens/declines while train accuracy keeps rising toward 1.0. This gap is called **overfitting** (or, described mathematically, the **bias-variance tradeoff**: as depth grows, the model's variance — its sensitivity to the specific training sample — increases faster than its bias decreases, so it starts fitting noise specific to the training set rather than the underlying pattern).

**Q7. What does the far-left end of the curve (depth 1-2) tell you?**
At depth 1-2 both train and test accuracy are lower (in the 70s%) and close together — that's **underfitting**, not overfitting. The mathematical reason a shallow tree has high bias: with only 1-2 splits, the tree can only carve the feature space into 2-4 regions total, which is far too coarse to capture the real decision boundary of an 8-feature dataset — it's forced to average over patients who actually have quite different outcomes, producing systematically inaccurate (biased) predictions on *both* train and test data equally.

**Q8. Which hyperparameter would you tune first to fix the overfitting shown in this plot?**
`max_depth`. It has the single most direct, most interpretable effect on model complexity for a decision tree, and this exact plot shows its effect isolated from every other hyperparameter — you can read the ideal value straight off the x-axis where test accuracy peaks. `min_samples_leaf` and `min_samples_split` help too, but their effect is less visually direct than "the tree can't grow past X levels."

### On the cost-complexity pruning path (Plot 4)

**Q9. What is the pruning algorithm trading off — impurity reduction vs. what?**
Impurity reduction vs. **tree complexity (number of leaves/splits)**. Each additional split has a "cost" in complexity; cost-complexity pruning only keeps a split if the impurity reduction it buys is worth more than `ccp_alpha` times the added complexity. As alpha rises, the bar for "worth it" rises, so more and more marginal splits get collapsed back into their parent leaf.

**Q10. Is the tree at the best alpha substantially smaller? What does that say about the unpruned tree's splits?**
Yes — dramatically so: **8 leaves** at the best alpha vs. **115 leaves** unpruned, a 93% reduction, while test accuracy actually *improved* (0.727 → 0.799). That tells you the overwhelming majority of the unpruned tree's splits (107 out of 115 leaves' worth) were memorizing noise specific to the training sample rather than capturing any real, generalizable signal in the data.

### On the confusion matrix (Plot 5)

**Q11. Which off-diagonal cell is larger, and which type of error matters more for this dataset?**
False positives (16) are slightly larger than false negatives (15) — they're close, but FP edges it out. For a diabetes screening context specifically, **false negatives (missing an actual diabetic patient) are usually the more costly error** in the real world — a missed diagnosis means a patient goes untreated, whereas a false positive just means extra follow-up testing that rules out diabetes. Given our matrix is nearly balanced between the two (16 vs 15), this pruned tree doesn't have an obviously dangerous bias in either direction, but if this were a real deployed screening tool, you'd want to bias the threshold toward reducing false negatives even further (see Q14).

**Q12. Is one class systematically confused with another, or is confusion spread evenly?**
The errors are fairly balanced (16 FP vs 15 FN out of 154), so there isn't a strong systematic bias toward one type of mistake here. That said, since the tree relies almost entirely on Glucose/BMI/Age, patients whose diabetes status doesn't line up with the "typical" pattern on those three features (like the one traced in Section 7 — high glucose but low BMI) are the ones most likely to land in the wrong leaf, regardless of which direction the error goes.

### On the ROC curve (Plot 6)

**Q13. How does a single tree's limited probability output show up on the ROC curve?**
A single tree can only output one probability value per leaf (the class proportion in that leaf), and this pruned tree has only 8 leaves — so it can only ever produce 8 distinct probability scores total. Visually, this means the ROC curve is **not smooth** — it's made of a small number of large, visible jumps/steps rather than the smooth curve you'd get from a model like logistic regression or a random forest that can output many finely-graded probabilities.

**Q14. Where does the default 0.5 threshold sit, and what would you sacrifice moving toward recall?**
The default threshold point (marked in red on the plot) sits reasonably close to the "elbow" of the curve, meaning it's already a fairly efficient trade-off point, not wastefully far from the top-left corner. If you cared more about recall (catching more true diabetics, fewer false negatives) than precision, you'd **lower the classification threshold** below 0.5 — meaning you'd label a patient "diabetic" even from a leaf with a lower confidence. The trade-off: you'd inevitably pick up more false positives (patients wrongly told they might have diabetes), sacrificing precision for recall.

### On the Gini feature importance plot (Plot 7)

**Q15. Is any top feature biased by having more possible split points, given it's a high-cardinality continuous feature?**
**Glucose** is exactly this kind of feature — it's continuous with many distinct values, so it offers the tree far more candidate thresholds to try than a low-cardinality feature would. That said, Glucose's dominance here (65.6% importance) is also strongly supported by domain knowledge (it's literally the primary clinical marker for diabetes), so its importance likely reflects genuine signal rather than pure cardinality bias — but the only rigorous way to verify that is to also compute **permutation importance** (shuffle each feature's values and measure how much accuracy drops) and check whether Glucose still dominates. Permutation importance doesn't have the same cardinality bias because it measures actual predictive contribution rather than how often a feature "won" a split.

**Q16. If you removed Glucose and retrained, would accuracy collapse, drop moderately, or barely move?**
It would most likely **drop moderately**, not collapse. A greedy top-down splitter doesn't give up when its favorite feature disappears — at the root, it would simply pick the next-best available split, which based on our importance ranking is almost certainly **BMI** (18.5% importance), since BMI and glucose are correlated with diabetes risk through related metabolic pathways. The resulting tree would likely be somewhat less accurate (glucose really is the strongest individual signal) but far from useless, because BMI and Age carry meaningful, if weaker, overlapping information.

### On the decision boundary plot (Plot 8)

**Q17. Why is the boundary blocky and axis-aligned, never diagonal or curved?**
Every single internal node in a decision tree tests **exactly one feature against one threshold** (e.g., "is Glucose > 154.5?"). A test like this can only ever draw a straight line that's perfectly vertical or perfectly horizontal in feature space — never diagonal — because it never combines two features into a single test (like "is Glucose + 2×BMI > 200?"). Since the final boundary is just the union of many such single-feature cuts stacked on top of each other, the result is always axis-aligned rectangles, no matter how many splits you add.

**Q18. If you nudged a boundary-edge point's features by 1%, could the prediction flip? What does that say about variance?**
Yes — a point sitting right on a boundary edge is, by definition, extremely close to a threshold, so a 1% nudge in the relevant feature could easily push it across the decision line and flip the predicted class entirely. This illustrates that a single decision tree is a **high-variance estimator**: small changes in input (or equivalently, small changes in the training data that shift where the threshold gets drawn) can produce disproportionately large changes in the prediction for points near a boundary. This is precisely the weakness that ensemble methods like Random Forests and Gradient Boosting are designed to fix, by averaging over many trees so no single threshold has that much influence.

---

## Summary

| Step | Result |
|---|---|
| Baseline (unpruned) | Train 1.000 / Test 0.727 — badly overfit, 115 leaves |
| Tuned via GridSearchCV | max_depth=3, min_samples_leaf=20 |
| Best pruning alpha | 0.00578 → 8 leaves |
| **Final pruned tree** | **Accuracy 0.799, F1 0.716, ROC AUC 0.804** |
| Top predictive features | Glucose (66%), BMI (18%), Age (12%) |

The core lesson of this pipeline: the fully-grown tree scored **worse** on new patients (0.727) than the aggressively pruned 8-leaf tree (0.799), despite the fully-grown tree being "more accurate" on the training data (1.000 vs ~0.85). Bigger, more complex trees are not automatically better — pruning traded away 107 of 115 leaves and *gained* accuracy, because most of that complexity was noise, not signal.
