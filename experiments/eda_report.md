# EDA Report — Amharic News Classification

## Train split

- Rows: **36804**
- Unique labels: **6**
- Empty/duplicate text rows dropped during cleaning (see src/data_utils.py::basic_clean)

| Label | Count | % of split |
|---|---|---|
| ሀገር አቀፍ ዜና | 14885 | 40.4% |
| ስፖርት | 7345 | 20.0% |
| ፖለቲካ | 6667 | 18.1% |
| ዓለም አቀፍ ዜና | 4684 | 12.7% |
| ቢዝነስ | 2773 | 7.5% |
| መዝናኛ | 450 | 1.2% |

**Class imbalance ratio (max/min): 33.08x** ⚠️ consider class weighting or macro-F1 as primary metric

**Text length (whitespace tokens):**

- mean: 250.0
- median: 180.0
- min: 1
- max: 6738
- 95th percentile: 680

_Use the 95th percentile as a sanity check for your tokenizer's max_length setting when fine-tuning Transformers._

Plots saved to `experiments/plots/`.

## Validation split

- Rows: **4090**
- Unique labels: **6**
- Empty/duplicate text rows dropped during cleaning (see src/data_utils.py::basic_clean)

| Label | Count | % of split |
|---|---|---|
| ሀገር አቀፍ ዜና | 1654 | 40.4% |
| ስፖርት | 816 | 20.0% |
| ፖለቲካ | 741 | 18.1% |
| ዓለም አቀፍ ዜና | 521 | 12.7% |
| ቢዝነስ | 308 | 7.5% |
| መዝናኛ | 50 | 1.2% |

**Class imbalance ratio (max/min): 33.08x** ⚠️ consider class weighting or macro-F1 as primary metric

**Text length (whitespace tokens):**

- mean: 248.7
- median: 180.0
- min: 1
- max: 3268
- 95th percentile: 681

_Use the 95th percentile as a sanity check for your tokenizer's max_length setting when fine-tuning Transformers._

## Test split

- Rows: **10205**
- Unique labels: **6**
- Empty/duplicate text rows dropped during cleaning (see src/data_utils.py::basic_clean)

| Label | Count | % of split |
|---|---|---|
| ሀገር አቀፍ ዜና | 4073 | 39.9% |
| ስፖርት | 2071 | 20.3% |
| ፖለቲካ | 1836 | 18.0% |
| ዓለም አቀፍ ዜና | 1314 | 12.9% |
| ቢዝነስ | 776 | 7.6% |
| መዝናኛ | 135 | 1.3% |

**Class imbalance ratio (max/min): 30.17x** ⚠️ consider class weighting or macro-F1 as primary metric

**Text length (whitespace tokens):**

- mean: 249.0
- median: 180.0
- min: 1
- max: 3327
- 95th percentile: 690

_Use the 95th percentile as a sanity check for your tokenizer's max_length setting when fine-tuning Transformers._


## Things to check manually before modeling

- [x] Skim 20-30 random articles per class — do labels look correct?
  - Spot checks surfaced a small number of Entertainment-labeled articles
    (e.g. voter conduct at polling stations, minimum driving-age
    requirements) that don't clearly match a conventional Entertainment
    definition — noted as a data quality observation in the report
    (Section 4.6) rather than corrected, since the true intended scope of
    the label couldn't be confirmed from available metadata.
- [ ] Check for near-duplicate articles (common in scraped news corpora)
- [ ] Check for any English-language leakage in the Amharic text
- [x] Confirm train/test label sets match exactly (no unseen labels in test)
  - Confirmed identical 6-label sets across train/validation/test via
    `data/prepare_local_data.py`'s validation check.

## Follow-up findings from later experiments

- Manual testing of the deployed model (see `app/app.py` and Section 7.3 of
  the report) found that Entertainment classification is sensitive to input
  length: a correctly-classified full-length article was misclassified when
  reduced to a single opening sentence lacking the music-specific vocabulary
  that appears later in the piece. This complements the imbalance finding
  above — Entertainment's small size (1.2% of train) combines with this
  length-sensitivity to make it the hardest category to classify reliably
  from short text specifically.