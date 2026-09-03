# Framing Index

A research pipeline for collecting comparable news coverage of the same events, measuring differences in how those events are framed, and evaluating a multi-feature Framing Index across news sources.

The project combines article collection, natural language processing, discourse-relation modeling, semantic comparison, and statistical validation. It is designed to compare articles covering the same event rather than treating framing as a property of an article in isolation.

## Overview

The pipeline has three main stages:

1. **Article collection and feature extraction** — `collect.py`
2. **Framing Index validation** — `validate.py`
3. **Results visualization** — `make_results_figures.py`

A separate setup script, `setup_models.py`, downloads the model checkpoint required for PDTB-style discourse-relation inference.

## Framing Features

The current Framing Index uses seven features:

- **Hedging** — measures differences in hedging language relative to other coverage of the event.
- **Emotional Amplification** — measures differences in emotional intensity and emotion distributions.
- **Discourse Emphasis** — measures differences in the prominence of terms and topics across coverage.
- **Omission / Informational Omission** — measures potentially missing salient information relative to the shared event-level information.
- **Semantic / Content Deviation** — measures how far an article's semantic content deviates from the event-level consensus.
- **Stance Misalignment** — measures differences in stance relative to the event and other coverage.
- **Causal Coherence** — measures differences in discourse-relation patterns using PDTB-style relation probabilities.

The current implementation combines the feature-level deviation scores into a consensus-adjusted Framing Index.

## Collection Modes

`collect.py` supports two approaches to constructing comparable event corpora.

### Fixed Events

In `fixed_events` mode, predefined event queries are used to identify relevant articles across configured news sources.

For each event, the pipeline:

1. searches configured RSS feeds,
2. scores candidate articles for event relevance,
3. extracts article text,
4. saves collection metadata,
5. computes the framing features, and
6. produces event-level framing datasets and rankings.

### Discovered Events

In `discovered_events` mode, the pipeline collects a broader pool of recent articles and discovers candidate events automatically.

Articles are:

1. collected from RSS feeds,
2. extracted and filtered,
3. embedded using a sentence-transformer model,
4. clustered according to semantic similarity,
5. evaluated using cluster-quality criteria, and
6. processed through the framing-feature pipeline when a cluster is accepted.

The collection strategy is selected in `collect.py`:

```python
COLLECTION_STRATEGY = "fixed_events"
# COLLECTION_STRATEGY = "discovered_events"
# COLLECTION_STRATEGY = "both"
```

## News Sources

The collection pipeline uses configured RSS feeds from multiple news organizations.

The current source configuration is defined directly in `collect.py`. Sources may be added, removed, or modified there as the corpus design develops.

## Project Structure

The main project files are:

```text
PythonProject4/
├── collect.py
├── validate.py
├── make_results_figures.py
├── setup_models.py
├── requirements.txt
├── .gitignore
├── discourse/
│   ├── __init__.py
│   └── infer_deberta_pdtb.py
└── multi_event_framing_project/
    ├── lexicons/
    ├── validation_outputs/
    └── event data and outputs
```

Large model checkpoints are intentionally excluded from Git and are installed separately.

The local `DP/` directory is a separate external repository and is not part of the main Framing Index codebase.

## Installation

The current development environment uses Python 3.14.

Create and activate a virtual environment, then install the project's Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` to reproduce the development environment.

## Required Lexicons

The feature-extraction pipeline expects the following resources:

```text
multi_event_framing_project/lexicons/NRC-VAD-Lexicon.txt
multi_event_framing_project/lexicons/NRC-Emotion-Lexicon.txt
```

These resources must be present before running `collect.py`.

## PDTB / DeBERTa Model Setup

Causal-coherence analysis uses a fine-tuned DeBERTa sequence-classification model for PDTB-style discourse relations.

The inference code is included in:

```text
discourse/infer_deberta_pdtb.py
```

The large model checkpoint is not stored in Git. Install it with:

```bash
python setup_models.py
```

The script downloads:

```text
config.json
pytorch_model.bin
```

and places them in:

```text
discourse/models/original_deberta_pdtb/checkpoint-1590/
```

The model predicts four high-level discourse-relation classes:

- Comparison
- Contingency
- Expansion
- Temporal

The tokenizer is loaded from `microsoft/deberta-v3-large`.

The model repository is currently hosted separately on Hugging Face. If that repository is private, authentication and permission to access it are required before running `setup_models.py`.

## Running the Pipeline

### 1. Collect Articles and Compute Framing Features

Run:

```bash
python collect.py
```

The desired collection strategy should first be selected using `COLLECTION_STRATEGY` in `collect.py`.

For successfully processed events, outputs include:

```text
metadata.csv
frozen_metadata.csv
framing_dataset.csv
proposal_labeled_features.csv
framing_rankings.csv
```

Extracted article text is also stored with the corresponding event data.

Discovered-event mode additionally produces discovery and cluster-quality outputs.

### 2. Validate the Framing Index

After event-level `framing_dataset.csv` files have been generated, run:

```bash
python validate.py
```

The validation pipeline combines available event datasets and performs statistical evaluation of the Framing Index and its component features.

The current validation procedure includes:

- descriptive statistics,
- synthetic criterion-score construction,
- Pearson correlation,
- Spearman correlation,
- feature-correlation analysis,
- source-level summaries,
- Elastic Net regression,
- cross-validated predictions,
- MAE, RMSE, and R² evaluation, and
- bootstrap coefficient stability analysis.

Validation outputs are stored under:

```text
multi_event_framing_project/validation_outputs/
```

Important outputs include:

```text
criterion_correlations.csv
source_summary.csv
feature_correlations.csv
elastic_net_predictions.csv
elastic_net_coefficients.csv
bootstrap_coefficients.csv
```

## Article Relevance Audit

Validation includes an article-level relevance audit:

```text
multi_event_framing_project/validation_outputs/article_relevance_audit_1.csv
```

This file is designed to preserve manual relevance judgments.

If manual `relevance_label` values are present, the validation pipeline can restrict analysis to articles labeled as relevant. If no manual labels have yet been entered, validation continues with the available unfiltered dataset.

Because this file contains human annotations, it should be preserved rather than treated as an automatically regenerable output.

## Generate Validation Figures

After running `validate.py`, generate the main results figures with:

```bash
python make_results_figures.py
```

The script reads the validation outputs and produces:

```text
figure_1_framing_index_vs_synthetic_score.png
figure_2_elastic_net_coefficients.png
figure_3_bootstrap_coefficients.png
figure_4_source_median_index.png
figure_5_feature_correlation_matrix.png
```

These figures are written to:

```text
multi_event_framing_project/validation_outputs/
```

## Complete Run Order

For a new environment, the intended sequence is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python setup_models.py
python collect.py
python validate.py
python make_results_figures.py
```

After the initial environment and model setup, subsequent analyses normally begin with `collect.py`.

## Main Outputs

At the event level, the project produces article metadata, extracted article text, feature-level framing measurements, overall Framing Index values, and article rankings.

At the validation level, it produces correlation analyses, model coefficients, cross-validated predictions, bootstrap estimates, source summaries, and publication-ready diagnostic figures.

## Reproducibility

The repository is structured so that machine-specific paths are not required for the main analysis workflow.

Python package versions are recorded in `requirements.txt`, and the large PDTB/DeBERTa checkpoint is downloaded separately using `setup_models.py`.

Large model weights, virtual environments, caches, and regenerable local artifacts should not be committed to Git.

Human-generated annotations, particularly the article relevance audit, should be retained.

## Research Status

This repository contains an actively developed research implementation. Feature definitions, validation procedures, event-selection methods, and corpus composition may continue to evolve as the methodology is evaluated.

Results should therefore be interpreted in the context of the specific code version, event corpus, feature definitions, and validation data used for a given analysis.