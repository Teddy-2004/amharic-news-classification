# Experiments Log

All six experiments below were run on the real dataset (36,804 train /
4,090 validation / 10,205 test examples after cleaning). Full discussion
and error analysis are in `report/Amharic_News_Classification_Report.pdf`,
Sections 6 and 7 — this log is the working record those sections are built
from.

| Run | Approach | Result (test split) | What I learned |
|---|---|---|---|
| baseline_lr | TF-IDF (ngram 1-1, max_features=50000) + Logistic Regression, class_weight="balanced" | acc=0.8289, macro-F1=0.8058 | Strong baseline. class_weight="balanced" gives Entertainment (1.2% of data) 96% recall at the cost of precision (0.65). Business is the weakest class (precision 0.54) — confirmed via confusion matrix to be mostly National News leaking in (362 of its false positives). |
| bilstm | BiLSTM (embed_dim=200, hidden_dim=128, layers=1), trained embeddings, unweighted | acc=0.7926, macro-F1=0.7071 | Underperforms the baseline by ~10pts macro-F1. Training loss drops to near-zero by epoch 10 while validation macro-F1 plateaus by epoch 3-4 — classic overfitting on 36,804 examples with from-scratch embeddings. Entertainment recall collapses to 0.36 without class weighting, confirming the baseline's "balanced" weighting was doing real work. |
| bilstm_weighted | BiLSTM, same architecture, class-weighted loss (inverse frequency) | acc=0.7909, macro-F1=0.7207 | Class weighting partially recovers Entertainment (F1 0.40→0.49) but doesn't close the gap to the baseline's 0.77 — weighting changes the loss penalty, not the model's ability to learn good representations from only ~450 training examples. Still trails the baseline overall by 8.5pts macro-F1, consistent with trained-from-scratch embeddings being data-hungry. |
| transformer_amharic_bert | Fine-tuned rasyosef/bert-medium-amharic (max_len=256, epochs=4, lr=2e-5), unweighted | acc=0.8732, macro-F1=0.8327 | **Best model overall.** Beats the baseline by 2.7pts macro-F1, BiLSTM by 11pts. Training curve shows steadily improving validation macro-F1 across all 4 epochs (unlike BiLSTM's plateau) — pretraining clearly provides better starting representations than learning embeddings from scratch on this amount of data. Took ~4 hours to train on CPU/MPS. |
| transformer_amharic_bert_weighted | Same model, class-weighted loss | acc=0.8581, macro-F1=0.8333 | Weighting trades precision for recall (macro-precision 0.8456→0.8214, macro-recall 0.8221→0.8543); net macro-F1 is essentially unchanged (+0.0006) while accuracy drops 1.5pts. Unlike BiLSTM, the pretrained model's representations didn't need loss-level rebalancing — recommend the **unweighted** version as the final/deployed model. |
| transformer_xlmr | Fine-tuned xlm-roberta-base (same hyperparameters), unweighted | acc=0.8194, macro-F1=0.7712 | Underperforms the Amharic-specific BERT by 6.15pts macro-F1 despite ~7x more parameters (1.12GB vs 162MB checkpoint) and ~2.6x longer training time (~10.4 hours vs ~4 hours). Loss curve shows slow initial adaptation (stays near 1.5 for ~65% of epoch 1 before meaningfully decreasing) — consistent with Amharic being a small fraction of XLM-R's ~100-language pretraining corpus, diluting the language-specific signal relative to a model pretrained on Amharic alone. |

## Final model selection

**Deployed model: `transformer_amharic_bert` (unweighted)** — best accuracy
and macro-F1, every per-class F1 score also the best among all six runs (not
just the average), and no precision/accuracy cost from unnecessary class
weighting. Published to the Hugging Face Hub at
`tedlai/amharic-news-classifier` and deployed via Gradio on Hugging Face
Spaces.

## Questions answered by these experiments

- **Did BiLSTM beat the TF-IDF baseline?** No — by ~8.5-10pts macro-F1
  depending on weighting. A from-scratch RNN with no pretraining struggles
  to beat simple sparse features on ~37k examples; TF-IDF's features need no
  "training" to be useful, while the BiLSTM's embeddings do.
- **Did the Transformer beat BiLSTM?** Yes, decisively — 11.2pts macro-F1,
  the largest gap between any two adjacent approaches tested. This is the
  clearest evidence in this project for the value of pretraining.
- **Language-specific vs. multilingual Transformer?** Language-specific
  wins by 6.15pts macro-F1, with a smaller model and less training time.
  Strong support for the choice to use an Amharic-specific pretrained model.
- **Which classes are hardest, and is it consistent across models?** Yes —
  Business and Entertainment have the lowest per-class F1 in every single
  model tested, while Sport is the easiest in every model (F1 ≥ 0.97
  throughout). See the report's Table 3 for the full cross-model breakdown.
- **Does class imbalance explain the hardest classes?** Partially.
  Entertainment's difficulty is well explained by its size (1.2% of
  training data). Business's difficulty is not primarily about size
  (7.5% of training data, larger than Entertainment) — confusion matrix
  analysis instead points to systematic confusion with the much larger
  National News category (see report Section 7.2).

## Hyperparameter variants tried

- TF-IDF: unigrams only (bigrams not tested — noted as a possible follow-up
  in the report's Limitations section).
- BiLSTM: single layer, embed_dim=200, hidden_dim=128 (not varied further
  given the architecture's weaker performance made further tuning lower
  priority than exploring the Transformer approaches).
- Transformer: 4 epochs, lr=2e-5, max_length=256 held constant across both
  Amharic BERT and XLM-R runs specifically so the model comparison wasn't
  confounded by differing training configurations. max_length=384 (informed
  by the EDA's ~680-token 95th percentile) was identified as a promising
  follow-up but not run, given time constraints — noted in the report's
  Limitations section.