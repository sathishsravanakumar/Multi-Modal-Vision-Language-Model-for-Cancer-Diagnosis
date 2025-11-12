# Multi-Modal-Vision-Language-Model-for-Cancer-Diagnosis

This repository contains Jupyter notebooks, datasets, and experiments for a project combining medical imaging and clinical text (clinical reports). The focus is on exploring single- and cross-modal models (vision and clinical-language models, plus cross-modal experiments) for tumor-related tasks.

## Repository layout (top-level)
- `Biobert.ipynb` — Notebook using BioBERT-style / BERT language models for clinical text experiments.
- `ClinicalBERT.ipynb` — ClinicalBERT experiments on clinical reports.
- `ViT.ipynb` — Vision Transformer experiments on medical images (if images are available or used in the workflow).
- `crossmodal_image_and_synthetic_text.ipynb` — Cross-modal training or data synthesis combining image features and synthetic text.
- `inference_crossmodal.ipynb` — Inference pipeline and examples for cross-modal model(s).
- `evaluate_crossmodal_model.ipynb` — Evaluation scripts and metrics for the cross-modal model.
- `train.csv`, `val.csv` — Original training and validation metadata / labels.
- `train_clean.csv`, `val_clean.csv` — Cleaned versions of the above CSVs (preprocessed).
- `clinical_reports.csv` — Raw clinical text dataset (reports).
- `README.md` — (this file)

## High-level goal / contract
- Inputs: clinical text CSVs (`clinical_reports.csv`, cleaned train/val CSVs), (optionally) medical images or derived visual features.
- Outputs: trained language, vision, and cross-modal models; evaluation metrics and plots; example inference outputs.
- Success criteria: reproducible notebook runs demonstrating training/inference with reproducible metrics (accuracy/F1/AUROC depending on task).

- ## Quick start (development environment)
1. Recommended Python
   - Python 3.8+ (3.10 is fine). GPU and CUDA are recommended for training vision models.
2. Create and activate a virtual environment (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```
3. Install typical dependencies used by the notebooks. There is no `requirements.txt` in the repo yet — add one for exact reproducibility. A minimal set:
```powershell
pip install jupyterlab notebook pandas numpy scikit-learn matplotlib seaborn torch torchvision transformers sentence-transformers timm
```
## Notebooks overview & tips
- `ClinicalBERT.ipynb` / `Biobert.ipynb`
  - Purpose: fine-tune/experiment with BERT-like models on clinical reports.
  - Key cells: data loading (from `clinical_reports.csv`), preprocessing, tokenization (Hugging Face `transformers`), model training loop or `Trainer` usage.
  - Tip: set `transformers` cache and `TORCH_HOME` to avoid re-downloading models repeatedly.
- `ViT.ipynb`
  - Purpose: vision-only experiments (Vision Transformer).
  - Key cells: dataset/dataloader, transforms, model, training loop.
  - Tip: verify images exist and that CSVs reference correct file paths.
- `crossmodal_image_and_synthetic_text.ipynb`
  - Purpose: combine image embeddings and synthetic/generated text for cross-modal training.
  - Tip: check the notebook for whether it expects precomputed image features or raw images.
- `inference_crossmodal.ipynb`
  - Purpose: show example inference flows (single-sample or batch inference).
- `evaluate_crossmodal_model.ipynb`
  - Purpose: compute metrics and visualizations. Typical metrics: accuracy, F1, precision/recall, AUC; use confusion matrices and PR curves for imbalanced problems.
