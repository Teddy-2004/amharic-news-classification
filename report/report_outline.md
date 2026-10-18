# Amharic News Topic Classification: A Comparative Study of Classical, Recurrent, and Transformer-Based Approaches

*A Summative Project in NLP for the African Context*

**Prepared by:** Tedla Tesfaye Godebo
**Course:** Machine Learning Technique I
**Submission Date:** October 18, 2026

**GitHub Repository:** https://github.com/Teddy-2004/amharic-news-classification.git
**Demo Video:** https://youtu.be/YGb_eK9PtCs
**Live System:** https://huggingface.co/spaces/tedlai/amharic-news-classifier

---

## 1. Abstract

This project aims to create and implement a topic classification system for Amharic news articles, which is one of the main challenges in the NLP community with respect to the lack of available, evaluated tools for the most widely used official language in Ethiopia. This paper applies and compares the following four different modelling strategies: baseline modelling using TF-IDF and Logistic Regression, Bidirectional LSTM learning trained using simple embeddings, Bidirectional LSTM learning using an Amharic-specific pretrained Transformer (bert-medium-amharic) with the same model trained using a multilingual Transformer (XLM-RoBERTa-base). The models are tested for the Amharic News Text Classification dataset consisting of 51,483 articles across six topic categories, and class-weighted loss variants are evaluated for the two recurrent models. Compared to other datasets, the dataset was highly imbalanced, with the largest class being 33 times the size of the smallest class, which had a significant effect on both the modeling strategy and the evaluation strategy, with macro-F1 being used as the primary measure in addition to the accuracy measure. The fine-tuned Amharic-specific Transformer showcased the highest overall accuracy (87.3%) and macro-F1 score (0.833) beating the multilingual Transformer by 6.2 macro-F1 points with a significantly lower number of parameters and the from-scratch BiLSTM by more than 11 macro-F1 points. The detailed error analysis showed that there was a common pattern for each of the four architectures: the "National News" category is a systematic overflow bucket, accepting most errors from all other categories, including 86.4% of all the errors involving the Entertainment category. Manual probing showed that the performance of Entertainment classification was sensitive to the length of the input, with short examples from source documents that were actually correctly classified being underperformed by longer examples. The final model was hosted as a public web application on Hugging Face Spaces using Gradio, so the users can classify Amharic text in real time.

## 2. Introduction

Amharic is the official working language of the Federal Democratic Republic of Ethiopia and one of the most widely spoken Semitic languages in the world, but is underrepresented in NLP research and development, compared to its number of speakers (Azime & Mohammed, 2021). The automatic topic classification of Amharic news text proves its practical importance for a number of downstream applications, like news content organization and search for Amharic content media platforms, article automatic routing for editorial workflows, media monitoring for Amharic language research and policy analysts, and as a core component of recommendation systems for Amharic language news readers.

The goal of this project is to build, test and operationalize an Amharic news article topic classification system that is not limited to one simple model. In particular, the project examines the following: (1) Whether classification performance scales as a function of a sequence of modeling techniques ranging from classical machine learning models to recurrent neural networks to language-specific pretrained models trained with fine-tuning; (2) Whether using language-specific pretrained models over a general-purpose multilingual model yields a measurable performance benefit on this task, and to what degree; (3) What systematic errors, if any, emerge across models, and what these errors tell us about the; (4) whether explicit class-weighting is the same across different models and if not, what it is and what it means for the category structure that lies behind the source data set.

The decision to select topic classification as the project's task, instead of machine translation or a domain-specific question answering system, is the result of careful project feasibility studies carried out in the project period. Compared to most of the available project options, Amharic has relatively matured labelled text classification materials, which means that most of the work can be done in modelling, experimentation and analysis without a significant amount of time being dedicated to data collection and annotation for the project's other allowed task types, which would have been required for this language.

## 3. Related Work

The data set in this project is derived from Amharic News Text Classification Dataset which was introduced by Azime and Mohammed (2021) and baseline results were achieved with classical machine learning approaches. They, as their work, are the source for Amharic news topic classification that this project is directly built on.

This and similar Amharic news corpora have been extended by subsequent work that enhances classification performance. The fine-tuning results of this project are compared to the results of fine-tuning models from the BERT family reported by Endalie (2025) for the Amharic multiclass news document categorization task. In more recent work, Taye et al. (2026) introduced a custom-built Amharic News Ontology along with TF-IDF and Logistic Regression, which achieved an accuracy of 97.0% on an 80/20 split of 61,915 articles from the Amharic News Dataset, an improvement of 3.2 percentage points over their baseline (TF-IDF). They also showed that the Amharic News Ontology reduced the confusion between Politics-Business by 38% specifically, which is attributed to the use of the ontology. Among others, their reported confusion pattern for Politics-Business is similar to a structurally related one found in this project: the errors in the categories Business and National News and Business are found in the same broad category; the errors in categories National News and Entertainment are not concentrated in this project, but are an unexpected surprise. This project does not build a handcrafted domain ontology but uses representation learning from pre-trained Transformers, making the comparison analogous to alternative approaches to handling the overlap between categories but not necessarily a direct comparison since the two projects have different splits and pre-processing of the data.

Adelani et al. (2023) presented MasakhaNEWS, a multi-language African-language news topic classification benchmark, which provided a basis for uniform methodology and evaluation conventions for the news topic classification task across all the languages, including the need for Amharic-specific evaluation work like this project. The models used for this project are selected from the Amharic-pretrained BERT variants publicly released on the Hugging Face Hub, which also offer an alternative language-specific pretraining to use for general multilingual models, such as XLM-RoBERTa (Conneau et al., 2020) used in this project as a comparison point in the experiments.

### Gap addressed by this project

Previous studies on this dataset usually report on one modelling technique. This work instead makes a controlled comparison between the four architectures using the same dataset, compares the effect of an explicit class-weighting on each architecture that can be trained, and performs a systematic qualitative error analysis that, to the best of this project's knowledge, has not been explicitly described in previous Amharic classification literature in the cross-category, cross-architecture fashion described here.

## 4. Dataset and Data Preparation

### 4.1 Source and Licensing

This project is based on Amharic News Text Classification Dataset by Azime and Mohammed (2021) and retrieved from a Hugging Face Hub mirror (rasyosef/amharic-news-category-classification) which allows preservation of original punctuation, whereas it was stripped in the original release. Sentence-final and word-separator marks (like ።) in Amharic do not have to be removed, as they can provide additional information that might be relevant for classical n-gram features and Transformer tokenizers and removing them without any clear justification will lose a signal that is potentially useful. This data is licensed under a CC-BY-4.0 license, and any use of the data in this project is allowed provided it is attributed.

### 4.2 Size, Splits, and Schema

The raw data set consists of 41,186 training examples, and 10,297 testing examples, with eight features each: The identifier, headline, category label, publication date, a viewcount field (pre-populated as "Unknown" for the vast majority of the examples, no longer used), the full article text, a source link, and a field containing a precomputed word length. Those columns not used as model features were retained in the working data and not used for modeling: article text, category label.

To support model selection and early stopping in the process of neural network training, a validation set was carved out of the training set (10%, stratified by label) to use for source release. There were 292 empty article text training rows dropped before splitting. The final splitted data sets for all experiments consisted of 36,804 training examples, 4,090 validation examples and 10,205 test examples after cleaning (empty-text rows removed and minor whitespace normalized, as described in section 4.4).

### 4.3 Class Imbalance and the Use of Sketches

The data is organized into six topic categories, which is native in Amharic, not translated into English. Table 1 shows the label distribution in the training split.

| Category (Amharic) | English gloss | Train count | % of train |
|---|---|---|---|
| ሀገር አቀፍ ዜና | National News | 14,885 | 40.4% |
| ስፖርት | Sport | 7,345 | 20.0% |
| ፖለቲካ | Politics | 6,667 | 18.1% |
| ዓለም አቀፍ ዜና | International News | 4,684 | 12.7% |
| ቢዝነስ | Business | 2,773 | 7.5% |
| መዝናኛ | Entertainment | 450 | 1.2% |

*Table 1. Category distribution in the training split (n = 36,804).*

The biggest ratio in the training split is 33.08x (National News: 14,885 examples, Entertainment: 450 examples) and 30.17x in the test split. This imbalance can have a significant impact on model performance when not addressed, and is an issue that is frequently tacked throughout this project's experiments, covered in detail later in this document, in section 5 and 6, through class-weighted loss functions and the adoption of macro-F1 as the key evaluation measure.

![Figure 1. Class balance in the training split, visualizing the data in Table 1.](../experiments/plots/class_balance.png)

### 4.4 Preprocessing

The preprocessing was intentionally kept to a minimum and documented in detail so that all the transformations that are applied to the raw text are documented. The applied steps were: (1) URLs were removed if they were present in the small number of articles as a side-effect, but were not related to the article subject, by matching them with a simple regular expression; (2) collapsed repeated spaces to single spaces; and (3) removed rows where the article text was empty after the above steps. The Amharic-specific punctuation has not been removed, but with the reasoning stated in section 4.1. There was a variation in tokenization strategy for every model, which is explained in section 5.

### 4.5 Text Length Characteristics

The length of articles (in terms of whitespace-separated tokens) varied widely, with a median value of 180, a mean value of 250.0 (with a right-skewed distribution and long tail of longer articles, up to a maximum of 6,738). The value of 680 tokens was the 95th percentile. The sequence window length for this distribution was also used to set the maximum sequence length during Transformer fine-tuning (section 5.3) as well as is discussed in section 7 as a contributing factor in Entertainment-category misclassification.

The number of whitespace tokens does not always reflect the number of words in Amharic because prefixes and suffixes are added to the word stems in this language, which is morphologically rich, and therefore may result in token counts that are different from the true word count by the Transformer models in this project. The numbers of counts that differ from the whitespaces-based numbers reported here. This is acknowledged as a limitation in section 9.

![Figure 2. Article length distribution (whitespace tokens), training split, with the 95th percentile marked.](../experiments/plots/text_length_hist.png)

### 4.6 Data Quality Observations

Exploratory data analysis led to a manual screening of a few articles, which revealed two concerns about quality of the data that warranted reporting openly without assuming the labels are necessarily correct. First, the following several articles, upon further examination, did not clearly fall under the scope of entertainment content—such as an article on the conduct of voters at polling stations and an article on the age at which people can drive legally in various countries. It is unclear from the metadata provided whether these are true examples of "label noise" from the original data, or whether a more expansive definition of "Entertainment" was used than is usually understood (such as editorial content that is a general human-interest or lifestyle category). Second, the labeling scheme used for this data set seems to be a very general one for National News, as opposed to a very specific one, which is investigated in detail in section 7 as a major result of the error analysis in this project; it was found to be the most frequent source of misclassification for all models tested in this project.

## 5. Methodology

Four models were employed and tested, with varying levels of complexity and representativeness: a baseline model, a recurrent neural network trained from scratch, a pre-trained language-specific Transformer model, and a pre-trained multilingual Transformer model. In the case of the two trainable neural architectures (BiLSTM and the Amharic Transformer), both unweighted and class-weighted loss variants were trained and compared, to directly measure the effect of addressing the class imbalance described in section 4.3.

### 5.1 Baseline: TF-IDF + Logistic Regression

The baseline model consists of TF-IDF weighted unigram features (n_max_gram=50000, min_df=2, term_freq is kept sublinear) and a classification with multinomial Logistic Regression. This configuration automatically used inverse-frequency class weighting (scikit-learn's "balanced" mode) due to the class imbalance issue that was documented in section 4.3. This method was selected because it is widely used in previous Amharic text classification studies as a benchmark method (Azime and Mohammed, 2021; Taye et al., 2026), it is considered as an interpretable method and has less computational requirements.

### 5.2 Bi-directional LSTM using trained embeddings

The second approach is to use a bidirectional LSTM (BiLSTM) trained with word embeddings from scratch (no pretraining), with whitespace tokenization and a vocabulary with a maximum number of tokens of 30,000 (with a minimum frequency of 2). The architecture consists of a single bidirectional LSTM layer (embedding dimension 200, hidden dimension 128 forward and 128 backward), followed by a 0.3 dropout head, followed by a linear classification head. Such an architecture was chosen as a representation of the recurrent neural network methods taught in the course, but to ensure that the contribution of the pretraining is isolated and that if the pretrained Transformer methods perform better, than this could be more clearly interpreted as an advantage of the pretraining, rather than an effect of the architecture alone.

An unweighted and class-weighted version were trained. The class weighted variant assigns weights to the cross entropy loss that are the reciprocal of the frequency of the classes: wc = N / (K · nc), where N is the number of training examples, K is the number of classes, and nc is the number of training examples in class c. This formula, which is the same as scikit-learn's "balanced" class weighting, assigns

It yielded a weight of about 13.63 for the Entertainment class, compared to 0.41 for the National News class, a difference of about 33.5 times the ratio of the underlying class imbalance, which was used for this data.

### 5.3 Fine-Tuned Amharic-Specific Transformer

The third approach is based on the Amharic pretraining of a BERT-architecture model, rasyosef/bert-medium-amharic, for the six-way classification task. The model's own subword tokenizer was used to tokenize input sequences and sequences were then truncated to a maximum length of 256 tokens, which was selected based on the 95th percentile of the whitespace-token length of the training data, but was not always so long; it was an approximation of 680 tokens (section 4.5); this truncation trade-off is discussed as a limitation in section 9. The fine-tuning process was performed for 4 epochs, learning rate of 2 × 10⁻⁵, batch size of 16, and early stopping was used based on the validation macro-F1 score and the patience was set to 2 epochs. Similarly, both unweighted and class-weighted loss variants were trained for the BiLSTM; the latter was done by subclassing the HuggingFace Trainer with a custom inverse-frequency weighting formula as described in section 5.2.

### 5.4 Fine-Tuned Multilingual Transformer

To ensure that the differences between the two models cannot be attributed to the different training regimes, the fourth approach uses xlm-roberta-base (Conneau et al., 2020), a Transformer pre-trained on the union of about 100 languages, and fine-tunes it with the same hyperparameters used for the Amharic model (section 5.3), while keeping the language-specificity of the pretraining corpora constant. Directly testing whether a language-specific pretrained model is the language-specific equivalent of a general-purpose multilingual model, this comparison directly tests the central research question of this project, which was aimed at methodologically justified modeling choices for African languages, as the Amharic language was put under the spotlight of this assignment.

### 5.5 Evaluation Metrics

All models are tested on the held-out test set and their accuracy, macro-averaged precision, macro-averaged recall, and macro-averaged F1 score are reported. Throughout this report, macro-F1 will be used as the main metric as accuracy on its own is misleading due to the class imbalance (33x) found in section 4.3. In addition to the per-class precision, recall, and F1, systematic patterns of errors are also studied through confusion matrices (sections 6 and 7).

## 6. Experiments and Results

Table 2 summarizes all six experimental runs conducted, reported on the held-out test split (n = 10,205).

| Run | Approach | Accuracy | Macro-F1 |
|---|---|---|---|
| Baseline | TF-IDF (unigram) + Logistic Regression, class-weighted | 82.89% | 0.8058 |
| BiLSTM | BiLSTM, trained embeddings, unweighted | 79.26% | 0.7071 |
| BiLSTM (weighted) | BiLSTM, trained embeddings, class-weighted | 79.09% | 0.7207 |
| Amharic BERT | Fine-tuned bert-medium-amharic, unweighted | 87.32% | 0.8327 |
| Amharic BERT (weighted) | Fine-tuned bert-medium-amharic, class-weighted | 85.81% | 0.8333 |
| XLM-R | Fine-tuned xlm-roberta-base, unweighted | 81.94% | 0.7712 |

*Table 2. Summary of all experimental runs on the held-out test split.*

Despite the model used for Amharic being much smaller (around 162 MB of parameters) than the multilingual model of XLM-R (around 1.12 GB of parameters) the unweighted fine-tuned Amharic BERT model outperforms both models (from-scratch BiLSTM and multilingual XLM-R) with a significantly better overall result (87.32% accuracy, 0.8327 macro-F1). This result directly informed the decision of going for a language-specific model, as explained in section 5.4, instead of a larger multilingual one.

### 6.1 Effect of Pretraining: BiLSTM versus Transformer

The difference between the best BiLSTM (0.7207 macro-F1) and the Amharic Transformer (0.8327 macro-F1, unweighted) is the greatest single difference among all consecutive methods in this study, 11.2 macro-F1 points. Even without considering the architectural difference alone, examining the training curves of the BiLSTM suggests that it is likely training loss is dropping because of a training set that is still small compared to the number of parameters that are being learned from scratch, including the word embeddings themselves, and that training loss had dropped from 0.85 to 0.03 over 10 epochs, with no sustained further drop after about epoch 3-4, whereas the validation macro-F1 had plateaued at that point and was swinging with no further sustained improvement after that point. The training curves of the finely-tuned Transformer, on the other hand, steadily rose in validation macro-F1 for all four training epochs (0.796, 0.816, 0.829, 0.832), indicating that the model is not only improving on the training data for this project, but that it is also benefiting from representations already learned during large scale Amharic pretraining.

### 6.2 Effect of Class-Weighted Loss

The two trainable neural architectures resulted in different class-weighted loss. However, for the BiLSTM, the increases of macro-F1 (1.36 points) and of Entertainment-class F1 (9.0%) were modest yet real: recall increased from 0.36 to 0.48, a factor of almost 2. The gains for the BiLSTM were modest but real: its macro-F1 score increased by 1.36 points, and its Entertainment-class F1 score increased by 9.0%, while the recall score increased by nearly 2 points (from 0.36 to 0.48). For the Amharic Transformer, on the other hand, the accuracy dropped by 1.5 percentage points (87.32% to 85.81%) while the left macro-F1 score remained almost the same (0.8327 to 0.8333, a difference of 0.0006). Understanding the underlying precision-recall trade-off explains why: class weighting raised macro-recall (0.8221 to 0.8543) while simultaneously lowering macro-precision (0.8456 to 0.8214), resulting in a small increase in the macro-F1 score, but a large decrease in overall accuracy, as the precision loss focused on the majority classes. It indicates that such possible advantages of class weighting are not generally applicable across architectures: It is beneficial for the rare-class representations learned by the under-trained BiLSTM, but has zero net gain over the already well-separated representations learned by the pretrained Transformer, in which it adds an unnecessary precision loss. The Amharic Transformer is thus deployed as the last model in this project, which is unweighted.

### 6.3 Effect of Pretraining-Corpus Specificity

The Amharic-specific Transformer achieved a higher macro-F1 score of 0.8327 compared to the multilingual Transformer whose score was 0.7712, with a similar fine-tuning hyperparameter setting. XLM-R's loss curve during training was significantly shallower during the initial adaptation, staying around 1.5 until the first two-thirds of the first training epoch, whereas the Amharic-specific model's loss started to drop significantly early on in training after just a few hundred steps. Even though both models were trained for the same number of epochs on the same data, XLM-R took some 10.4 hours of wall clock training time, whereas the Amharic-specific model took about 4 hours of wall clock training time for the same hardware and batch size. The loss convergence is slower, and the training time is longer, as is typical when using a small fraction of the language's text, Amharic, in the approximately 100-language pretraining set used for XLM-R.

## 7. Error Analysis and Discussion

### 7.1 Per-Class Performance Across Architectures

Table 3 compares per-class F1 scores across the baseline, the better-performing BiLSTM variant (class-weighted), the best Transformer (Amharic BERT, unweighted), and XLM-R.

| Category | Baseline | BiLSTM (w) | Amharic BERT | XLM-R |
|---|---|---|---|---|
| National News | 0.82 | 0.79 | 0.87 | 0.82 |
| Entertainment | 0.77 | 0.49 | 0.73 | 0.63 |
| Sport | 0.98 | 0.97 | 0.99 | 0.98 |
| Business | 0.64 | 0.57 | 0.69 | 0.64 |
| International | 0.86 | 0.80 | 0.90 | 0.83 |
| Politics | 0.76 | 0.70 | 0.81 | 0.74 |

*Table 3. Per-class F1 score by model.*

There are two patterns that are consistent with all of the models tested. Sport and Entertainment have the highest and most consistent performance on all four approaches and Sport in particular has vocabulary (player terminology, match terminology for Sport; music, film and performance terminology for Entertainment, when available) that is lexically distinct from the other four categories. Second, there is a group of consistently low and variable F1 scores across all architectures, that are consistently found in National News, Politics and Business, compared to Sport or International News. Finally, the Amharic BERT model performs best in each individual category, and not just in the average case, thereby reinforcing the result in section 6.3 that it has an actual representation improvement, and not just a trade-off in select categories.

### 7.2 National News as a Systematic Overflow Category

The confusion matrix analysis of the baseline model reveals that National News is the category with the highest rate of misclassification (1037/4073 = 25.5% of the test set examples are misclassified in this category, most of them as Politics (402 cases) and Business (362 cases), while the opposite direction is much weaker (77 articles from Business are misclassified as National News). This asymmetry is captured by a large number of National-News-to-Politics errors in the former category, and a smaller number in the latter, for all four models tested, the Amharic BERT model showing a smaller difference (256 errors to 402 errors for base line, and 502 errors for XLM-R) but not a complete elimination of the trend.

In addition to the three above categories that are in topical proximity, this trend also holds true of Entertainment, a category that has no obvious topical connection to National News. The Amharic BERT model misclassified 44 Entertainment articles, 38 of which (86.4%) were classified as National News while the remaining 6 were categorized under all other categories. In contrast, this was not a hypothesis that was set out to be tested, but was a result that came out of the systematic analysis of the misclassified-example output.

Overall, the results suggest a unified interpretation: National News is not a specific topic, but it is a default, general purpose category that the model tends to fall back on when the lexical signal of the topic is weak, ambiguous, or nonexistent, both when the true topic of the article is topically related (such as Politics or Business) and when it is not (such as Entertainment). This interpretation follows in the footsteps of Taye et al. (2026), who reported a similar phenomenon, the Politics-Business confusion pattern, for a similar news set written in Amharic; and indicates that the category overlap problem in Amharic news classification is not necessarily a pairwise category confusion problem, but a general overflow-category problem.

### 7.3 Entertainment Classification Is Sensitive to Input Length

A handwritten single sentence example "entertainment" with the description of a movie being released at a cinema was confidently (97% confidence) classified as "sport" during manual testing of the deployed application (section 8). The result was investigated and the Amharic verb ይመረቃል (transliterated "yìmäräqal", approximately "will be inaugurated"), which appeared in the test sentence, was identified as a potential contributing factor: News in Amharic is often associated with the opening of infrastructure/ facilities including stadiums, and a follow-up test sentence describing a stadium inauguration, genuinely Sport-related content, using the same verb was correctly classified with 100% confidence as Sport. This is in line with the model being trained to create an association between this verb and the category of Sport strong enough to override the other clear indicators of Entertainment that are present in the first test sentence ("film", "cinema").

This sensitivity was further characterized in an independent test. When tested alone, the same model misclassified one sentence from an authentic test-set article correctly classified by the model as Entertainment (a 1,311 character profile of a musician). The model could correctly classify the full article as Entertainment when the music-specific vocabulary (music, cassette/album, guitar, singer) was not included in the extracted sentence, but was seen later in the full article. This means that the Entertainment classification is based not on any single sentence in the article, but on the overall combination of vocabulary that appears throughout the text, a characteristic not captured by the aggregate metrics alone, and directly relevant to the reliability of the model on short-form texts, like headlines, social media posts or excerpts from articles (discussed in further detail as a limitation in section 9).

### 7.4 Confusion Matrices for All Models

The full confusion matrices for all five model configurations trained here are shown in Figures 3 to 7. For space reasons, only one variant of each particular model is shown here; the class-weighted Amharic BERT and class-weighted BiLSTM are sufficiently similar in patterns to their respective unweighted variants that only one variant of each is presented here, with the unweighted BiLSTM presented to illustrate the overfitting-related error patterns discussed in section 6.1, while the weighted BiLSTM is presented as used in the cross-model comparison in Table 3. Each row is a true class and each column is a predicted class; cells on the diagonal are cells that have been predicted correctly.

![Figure 3. Confusion matrix — TF-IDF + Logistic Regression baseline (class-weighted)](../experiments/plots/baseline_lr_confusion_matrix.png)

![Figure 4. Confusion matrix — BiLSTM (unweighted).](../experiments/plots/bilstm_confusion_matrix.png)

![Figure 5. Confusion matrix — BiLSTM (class-weighted).](../experiments/plots/bilstm_weighted_confusion_matrix.png)

![Figure 6. Confusion matrix — fine-tuned Amharic BERT-medium (unweighted, final deployed model).](../experiments/plots/transformer_rasyosef__bert-medium-amharic_confusion_matrix.png)

![Figure 7. Confusion matrix — fine-tuned XLM-RoBERTa-base (unweighted).](../experiments/plots/transformer_xlm-roberta-base_confusion_matrix.png)

## 8. Deployment

The unweighted fine-tuned Amharic BERT-medium model (selected as per the reasoning in section 6.2) was deployed as an interactive web application using Gradio and deployed on Hugging Face Spaces. The app receives text from the Amharic Language (either a headline or a complete article) and gives the predicted category and the entire probability distribution across all six categories, so that the user can be aware of the top predicted category as well as its relative confidence compared to the other related categories.

The trained model checkpoint was uploaded to the Hugging Face Hub as a standalone model repository, and the application starts up and loads the model directly from the Hugging Face Hub, instead of being put in the application repository, which then gets committed into the Space's git history, with a big binary file. The application code uses the ZeroGPU-required allocation decorator on its inference function with the Space running on Hugging Face's free ZeroGPU tier, which allocates GPU compute dynamically per inference call; this has no impact on application code outside of the Spaces environment, as it is fully functional without the decorator.

Prior to deployment, example inputs were packaged with the application interface and each input was checked against the live model's actual predictions, which uncovered the length sensitivity discovered in the Entertainment category as reported in section 7.3 and led to the final selection of a correctly-classified full-length article from this category as the application's Entertainment example.

## 9. Limitations and Future Work

- **Class imbalance remains only partially addressed.** The worst per-class F1 scores, 0.69 and 0.73 respectively, are seen even in the best performing model on minority classes Business and Entertainment. Further work may focus on data augmentation on these categories, attempting to use a focal loss function instead of simply inverse-frequency weighting, or active collection of further minority class examples.
- **The National-News overflow pattern** is an example of a labeling characteristic that this project's choice of models is not able to fully account for in the source dataset. If this were pursued further, a new taxonomy for categories would be needed (e.g., breaking the National News category into more refined subcategories) or a recategorization of the source data (this is not part of this project).
- **Entertainment classification is sensitive to input length** (section 7.3), and therefore the reliability of the model on shorter input lengths, e.g. headlines or social media posts, might be lower than its presented test set results, which are based on longer inputs, e.g. full-length articles. This is a significant precondition in any practical application where the task is to classify short texts.
- **Maximum sequence length (256 tokens) truncates a meaningful share of articles**, given a 95th-percentile article length of approximately 680 whitespace tokens (section 4.5), and whitespace tokenization itself imperfectly approximates true word count for Amharic given its morphological complexity. Multiple maximum sequence lengths (256 vs. 384 vs. 512 tokens) were identified as a promising follow-up experiment but not completed within the time frame of this project.
- **No hyperparameter search was conducted** beyond the configurations reported in section 5; learning rate, batch size, and number of training epochs were set to reasonable defaults for Transformer fine-tuning rather than tuned systematically, given the computational cost of each training run (up to approximately 10.4 hours for XLM-R on the available hardware).
- **Evaluation is conducted using a single train/validation/test split**, as it is typically done when training a Transformer for fine-tuning due to its computational cost, but reported metrics have a certain unquantifiable variance not captured by a single train/validation/test split.
- **Future work** could involve expanding this project to include the larger, more recent Amharic news dataset (covering articles through late 2024, rather than 2021) to include the Taye et al. (2026) Ontology-based feature fusion approach for comparison or to add the model confidence score as a cue for the end user that a prediction may not be reliable.

## 10. Conclusion

This project involved the development, evaluation and deployment of a system for Amharic news topic classification that covers four modeling approaches of increasing complexity. There are three main empirical findings that are central. First, the results show that the pretrained Amharic-specific Transformer clearly and consistently outperforms the from-scratch BiLSTM by 11 macro-F1 points and the multilingual Transformer with a similar fine-tuning configuration by 6 macro-F1 points, suggesting that using language-specific pretraining is definitely more useful than a larger, more general-purpose pre-training configuration on this task. Third, class-weighted loss is not generally superior for all architectures: while it did make a significant difference in the performance of the under-trained BiLSTM on the very under-represented Entertainment class, it resulted in no net macro-F1 benefit, and a measurable macro-accuracy loss, for the already well-trained Transformer, suggesting the value of explicit class-weighting depends on how much of a model's under-represented-class performance gap is due to representation quality versus loss-function bias. Third, as far as systematic errors across all four architectures are concerned, the dataset's National News category seems to be a category in which there is no particular pattern of error, but rather the largest proportion of misclassified data from all other categories, many of which are topically unrelated, such as the topically unrelated Entertainment category, and only explainable in terms of the label structure itself rather than a model shortcoming.

The most surprising and captivating result of this project, however, is that the overall classification of the articles was, in practice, more dependent on the amount of the vocabulary that appeared across article rather than on any individual sentence, and that it was possible to demonstrate this by simply taking a real test-set article, removing a category-indicating word from it, and then re-classifying it; as seen in the actual test-set article that was correctly classified when the words of the entire article were used but incorrectly when the words of only a single sentence were used. This is an example of the usefulness of the qualitative probe done at the example level rather than the test-set level, and would not be apparent on just the test-set metrics.

## 11. References

Adelani, D. I., Neubig, G., Ruder, S., Rijhwani, S., Beukman, M., Palen-Michel, C., Lignos, C., Alabi, J. O., Muhammad, S. H., Nabende, P., ... & Dossou, B. F. P. (2023). MasakhaNEWS: News topic classification for African languages. *Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*. arXiv:2304.09972.

Azime, I. A., & Mohammed, N. (2021). An Amharic news text classification dataset. arXiv:2103.05639.

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 8440-8451.

Endalie, D. (2025). Fine-tuning BERT models for multiclass Amharic news document categorization. *Complexity*. Wiley.

rasyosef. (n.d.). *bert-amharic: BERT transformer models pretrained on Amharic text* [Software repository]. GitHub. https://github.com/rasyosef/bert-amharic

rasyosef. (n.d.). *amharic-news-category-classification* [Dataset]. Hugging Face Hub. https://huggingface.co/datasets/rasyosef/amharic-news-category-classification

Taye, B. G., Desta, A. B., & Alene, A. Y. (2026). Enhancing Amharic news classification through ontology-based feature fusion and logistic regression. *Scientific Reports*.