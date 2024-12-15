# NBME - Score Clinical Patient Notes

## Overview

This project focuses on identifying key phrases in patient notes from medical licensing exams. The goal is to develop an automated method that maps clinical concepts (referred to as "Features") from exam scoring criteria, such as "loss of appetite," to the various ways these concepts are expressed in medical students' patient notes, like "eating less" or "looser clothes."

**Competition Page**: [NBME - Score Clinical Patient Notes | Kaggle](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/)

---

## Data Description

The dataset consists of approximately 40,000 patient notes belonging to 10 clinical cases. However, only 1,000 notes have feature annotations. The test set includes around 2,000 annotated notes. The key to achieving high performance in this competition is leveraging the 30,000+ unannotated notes using unsupervised learning.

### Key Terms
- **Clinical Case**: Type of case. There are 10 cases in this dataset.
- **Patient Note**: Notes written by medical students during consultations.
- **Feature**: Clinical concepts that need to be predicted and mapped to relevant expressions in patient notes.

### Files
- **patient_notes.csv**: Contains unique identifiers, clinical cases, and note texts.
- **features.csv**: Describes the features with unique IDs and case associations.
- **train.csv**: Contains annotated data mapping features to specific note expressions.

---

## Solution Approach

### Steps
1. **Pretraining**:
   - Pretrained DeBERTa models (`microsoft/deberta-base`, `microsoft/deberta-large`, `microsoft/deberta-v3-large`) using all note data (with and without annotations).

2. **Training**:
   - Used labeled data (14,000 samples, including 1,000 patient notes) for 5-fold training.

3. **Pseudo Labeling**:
   - Generated pseudo labels for unlabeled notes using trained models. Used blending techniques to combine predictions from the three models with weights of `base:0.19`, `large:0.37`, `v3-large:0.44`.

4. **Training with All Data**:
   - Combined labeled and pseudo-labeled data to further train the models.

5. **Fine-tuning**:
   - Fine-tuned the models on labeled data.

6. **Inference**:
   - Used all model weights for final predictions. Blending weights for inference: `base:0.15`, `large:0.25`, `v3-large:0.6`.

---

## Code and Models

### Code Files
1. `nbme-pretrain.ipynb`
2. `nbme-train.ipynb`
3. `nbme-pseudo-prediction.ipynb`
4. `nbme-pseudo-blend.ipynb`
5. `nbme-alldata-train.ipynb`
6. `nbme-finetune.ipynb`
7. `nbme-inference.ipynb`

### Datasets
- `train_processed.pkl`: Labeled dataset.
- `train_pl_all.pkl`: Full dataset (labeled + pseudo-labeled).

---

## Results

- **DeBERTa-base**: Public LB: 0.868
- **DeBERTa-large**: Public LB: 0.873
- **DeBERTa-v3-large**: Public LB: 0.878
- **Final Blend**: Private LB: 0.883 (Top 1%)

---

## TL;DR

The goal of the competition is to map clinical concepts from scoring criteria to expressions in patient notes. The approach includes fixing noisy labels, leveraging pseudo-labels for unlabeled data, and using DeBERTa models (`base`, `large`, and `v3-large`) with 5-fold cross-validation and blending techniques. This approach achieved a Top 1% score on the leaderboard.
