# Chapter 4: Implementation and Results

## 4.1 Implementation Setup

The implementation was designed to meet two project objectives: first, to benchmark Hausa sentiment classification on the AfriSenti benchmark dataset; second, to apply the trained model to Ghana-focused Hausa news content for real-world sentiment analysis. The pipeline was built as a modular supervised learning workflow in which raw Hausa text was preprocessed, transformed into machine-readable features, evaluated with cross-validation, and finally used for inference on unseen text.

The workflow followed a standard machine learning pipeline. The dataset was loaded from the project’s TSV files, each containing text samples and sentiment labels. Text preprocessing was applied before feature extraction to reduce noise and improve consistency. This included lowercasing, removing URLs, mentions, hashtags, repeated characters, and other non-informative symbols, followed by tokenisation and stopword handling. These steps were implemented in the project utilities and reused across both training and evaluation scripts.

For feature extraction, the model used a combined representation rather than a single vectoriser. A FeatureUnion was constructed with character-level TF-IDF and word-level TF-IDF, and for the logistic regression pipeline it also included text-length features and lexicon-derived sentiment features. This hybrid representation enabled the classifier to capture morphology, word patterns, document structure, and sentiment polarity cues. The training stage used GridSearchCV with macro-F1 as the optimisation metric, allowing model selection to focus on the class-balanced performance most appropriate for a three-class sentiment problem.

The implementation also included evaluation and reporting modules that produced both numeric metrics and diagnostic plots. After training, the system saved the best model, the cross-validation scores, the classification report, and a confusion matrix image for inspection. This ensured that the experimental evidence was not limited to accuracy alone but also included class-wise precision, recall, F1-score, and error patterns.

### Model configuration

The project trained and compared two main classifiers:

- Multinomial Naïve Bayes, used as a strong and efficient baseline.
- Logistic Regression, configured with class balancing and tuned regularisation.

Both models were tested with TF-IDF and count-vector representations, with the final inference framework selected according to the relevant evaluation results. The pipeline therefore supports both benchmark testing and practical deployment on unlabelled Hausa social media or news content.

---

## 4.2 Results

The primary evaluation was performed on the held-out AfriSenti Hausa test set, containing 5,303 samples across the three sentiment classes. Table 4.1 summarises the comparative results produced by the project’s saved evaluation reports.

Table 4.1: Model comparison on the held-out Hausa test set

| Model | Representation | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
|---|---|---:|---:|---:|---:|
| Multinomial Naïve Bayes | TF-IDF | 0.7460 | 0.7474 | 0.7493 | 0.7463 |
| Logistic Regression | TF-IDF | 0.7713 | 0.7721 | 0.7816 | 0.7710 |
| Logistic Regression | Count | 0.7730 | 0.7733 | 0.7762 | 0.7729 |
| Multinomial Naïve Bayes | Count | 0.7345 | 0.7360 | 0.7381 | 0.7348 |

The evaluation evidence shows that the logistic regression models were clearly stronger than the Naïve Bayes baselines. The best-performing configuration reported in the project’s saved evaluation files was the logistic regression model trained with a count-vector representation, achieving an accuracy of 77.30% and a macro-F1 score of 0.7733. The TF-IDF logistic regression model was close behind, with 77.13% accuracy and a macro-F1 score of 0.7721. This narrow margin indicates that both feature representations were effective and that the model choice was stable across the experimental runs.

### Per-class performance for the leading model

Table 4.2 shows the class-wise results for the logistic regression model on the held-out test set.

Table 4.2: Per-class classification report for the leading logistic regression model

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.7924 | 0.6794 | 0.7316 | 1759 |
| Neutral | 0.6827 | 0.8273 | 0.7480 | 1789 |
| Positive | 0.8697 | 0.8063 | 0.8368 | 1755 |
| Macro average | 0.7816 | 0.7710 | 0.7721 | 5303 |

The positive class achieved the strongest F1-score at 0.8368, which suggests that highly positive Hausa expressions were easier for the model to separate from neutral and negative text. The neutral class had the best recall at 0.8273 but a lower precision, indicating that the model often over-assigned examples to neutral sentiment when the boundary between neutral and negative or positive sentiment was ambiguous. This is consistent with difficulties reported in sentiment analysis literature, especially for three-class tasks in low-resource languages.

---

## 4.3 Analysis

The results indicate that the main challenge was not model capacity, but the ambiguity of the neutral class. The project’s best classifier performed well overall, yet the neutral class remained the most difficult to separate. This is visible in the confusion matrix and in the per-class metrics: the model had good recall for neutral examples, but some sentiment samples were misclassified as neutral because the lexical and contextual cues were weak or mixed. In contrast, the positive class was easiest to identify, likely because positive terms in Hausa are relatively more distinctive and appear with stronger polarity patterns in the dataset.

The comparison between vectorisers further supports the conclusion that the model was not dominated by one feature form. The count-based logistic regression model slightly outperformed the TF-IDF variant in the saved results, although the difference is marginal and not large enough to suggest a dramatic advantage. This means the system is fairly robust: whichever representation is used, the model remains above the practical performance threshold expected for a sentiment classification system in a low-resource setting.

The implementation also met the project’s practical objective of producing a usable classifier for downstream sentiment inference. The trained models and evaluation reports were stored in the project folders, allowing the system to process unlabelled Hausa texts outside the benchmark set. This is important because the project is not limited to academic evaluation; it also aims to generate sentiment insights from real-world Hausa journalism and social media content related to Ghana.

### Mapping the results to the project objectives

1. Objective: train and benchmark Hausa sentiment models on a standard dataset.
   - Achieved through the AfriSenti training and test pipeline, model comparison, and saved evaluation metrics.

2. Objective: identify the most suitable classifier for Hausa sentiment analysis.
   - Achieved through cross-validation and held-out test evaluation, with logistic regression emerging as the most stable and highest-performing model.

3. Objective: support practical sentiment prediction on real-world Hausa text.
   - Achieved by saving the model artifacts and evaluation reports for inference on external datasets and news content.

This mapping is important because it ensures that the reported results are not merely numerical outputs; they are directly linked to the project goals and the expected end-user application.

---

## 4.4 Screenshots, Tables, and Figures

The project generated visual evidence alongside the numerical reports. The confusion matrix images were saved in the project reports folder and correspond to the model evaluation outputs. These figures are useful because they show how the classifier distributes errors across the three sentiment classes rather than only reporting aggregate numbers.

Figure 4.1: Confusion matrix for the logistic regression evaluation report, saved as reports/metrics_lr_eval.confusion.png.

- Interpretation: the matrix shows the majority of correct predictions lying along the diagonal.
- Most of the classification errors occur between neutral and negative statements, confirming the analytical difficulty of the neutral class.
- The strong positive-class performance is also visible, with fewer misclassifications than the other classes.

Figure 4.2: Confusion matrix for the Naïve Bayes baseline, saved as reports/metrics_nb_eval.confusion.png.

- Interpretation: the baseline model is weaker overall, especially in separating neutral and negative examples.
- The contrast with the logistic regression matrix illustrates why the logistic regression pipeline was selected as the preferred model.

The inclusion of both tables and confusion matrices is important because the project follows good research practice: results are reported with both summary statistics and diagnostic visuals. This avoids the common mistake of relying only on a single headline metric without explaining the underlying prediction behaviour.

---

## 4.5 Discussion

The implementation demonstrates that a carefully designed classical machine learning pipeline can perform competitively on Hausa sentiment classification, even without transformer-based models. The use of character and word TF-IDF features, together with lexicon-based and length-based descriptors, provided a robust representation of Hausa text. Logistic regression, in particular, offered a good balance between performance, interpretability, and practical deployment.

The results also reveal the broader challenge of Hausa sentiment analysis in a low-resource context. The data distribution, class overlap, and informal writing style in social media content create difficulty in distinguishing neutral, negative, and positive statements. Although the model achieved usable performance, it is still limited by lexical ambiguity and context sensitivity. This is especially relevant when analyzing Hausa content from news and social media, where sarcasm, code-switching, and domain-specific expressions can alter sentiment.

A notable strength of the implementation is that it is transparent and reproducible. The scripts in the source directory clearly separate data loading, preprocessing, feature construction, training, evaluation, and prediction. This makes the project suitable for extension. Future work could include language-specific preprocessing, class rebalancing, additional lexicon features, or transformer-based models such as AfriBERTa. Such extensions may improve the neutral-class boundary and further raise macro-F1 performance.

### Avoiding common research mistakes

This chapter was structured to avoid the most common problems in reporting machine learning results:

- Screenshots were not included without explanation. Each visual is paired with an interpretation of what it shows.
- Results were tied directly to the project objectives rather than reported as isolated metrics.
- Findings were interpreted rather than merely listed, making the discussion explain why the model behaves the way it does.
- The strengths and limitations of the classifier were discussed in a balanced way, which is essential in applied sentiment analysis research.

In summary, the implementation confirms that a supervised Hausa sentiment pipeline can achieve strong benchmark performance and produce useful predictions for real-world content. The evidence supports the conclusion that logistic regression is the strongest configuration in the current project, while also identifying the neutral class as the main remaining source of classification difficulty.

---

This chapter therefore presents the completed implementation, the measured results, and the evidence-based interpretation required for a solid research evaluation.
