# Framing Index

A research pipeline for collecting news coverage of the same events, measuring differences in how the events are framed using a myriad of features and evaluating a resulting Framing Index incorporating a multitude of news sources. This is WIP.

The project encompasses information retrieval, NLP, discourse-relation modeling, lexicon and semantic comparison, and feature validation methods.

## Overview

The pipeline is constituted of three stages:

1. **Information retrieval/Article collection and feature extraction** — `collect.py`
2. **Feature and Index validation** — `validate.py`
3. **Results** — `make_results_figures.py`

To complement these files, a setup script file, `setup_models.py`, is provided with the downloadable model checkpoint required for the PDTB-style discourse-relation feature ('Causal Coherence' feature).

## Features

The current iteration of the Framing Index uses seven features:

- **Hedging** — measures differences in hedging language in an article relative to overall coverage of the event.
- **Emotional Amplification** — measures differences in an article's emotional intensity relative to the emotion distribution from the overall coverage.
- **Discourse Emphasis** — measures differences in the prominence of terms across coverage relative to the average of the event's corpora.
- **Omission** — measures potentially missing salient information in an article relative to event-level corpora.
- **Semantic / Content Deviation** — measures how far an article's semantic content deviates from the event-level consensus.[Can I put this under Omission?]
- **Stance Misalignment** — measures difference in stance of an article relative to average event-level coverage.
- **Causal Coherence** — measures differences in discourse-relation patterns using PDTB-style relation probabilities.
- **Causal Coherence** — measures differences in discourse-relation patterns using PDTB-style relation probabilities.

This implementation combines the feature-level deviation scores into a Framing Index score.

## Collection

`collect.py` supports two WIP approaches to constructing comparable event corpora - Fixed events and Discovered Events.

### Fixed Events

The `fixed_events` approach requires predefined events. Queries are then used to identify relevant articles across a given set of news sources.

Given any event, the pipeline is as shown:

1. searches RSS feeds,
2. scores candidate articles for event relevance,
3. extracts article text,
4. saves collection metadata,
5. computes the framing features,
6. produces event-level framing datasets and rankings.

### Discovered Events

The `discovered_events` approach identifies candidate events automatically by analyzing and sorting through a large pool of articles.

Articles are:

1. collected from RSS feeds,
2. extracted and filtered,
3. embedded using a sentence-transformer model,
4. clustered according to semantic similarity,
5. evaluated using cluster-quality criteria, and
6. processed through the framing-feature pipeline when a cluster is accepted.

The current default is 'Fixed Events' as this is a WIP.

```python
COLLECTION_STRATEGY = "fixed_events"
# COLLECTION_STRATEGY = "discovered_events"
# COLLECTION_STRATEGY = "both"
```

## News Sources

The collection pipeline uses RSS feeds from multiple news organizations. Sources may be added, removed, or modified in 'collect.py' as the corpus progresses.

## Project Structure

The project files are as shown:

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

Large model checkpoints have been intentionally excluded from git. The local `DP/` directory is a separate external repository and is not part of the main Framing Index codebase.

## Installation

The current development environment uses Python 3.14.

Instruction: (1) Create a virtual environment then (2) Install Python dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt`. Using this, one can reproduce the required development environment.

## Required Lexicons

The feature-extraction pipeline expects the following '.txt' files:

```text
multi_event_framing_project/lexicons/NRC-VAD-Lexicon.txt
multi_event_framing_project/lexicons/NRC-Emotion-Lexicon.txt
```

These files must be present and available before running `collect.py`.

## PDTB / DeBERTa Model Setup

Causal-coherence analysis uses a fine-tuned DeBERTa sequence-classification model for PDTB-style discourse relations.

The code is found in:

```text
discourse/infer_deberta_pdtb.py
```

The large model checkpoint (1590) is not stored in git. Instead, Install it with,

```bash
python setup_models.py
```

Running the file downloads:

```text
config.json
pytorch_model.bin
```

and puts them in,

```text
discourse/models/original_deberta_pdtb/checkpoint-1590/
```

The model predicts four high-level discourse-relation classes:

- Comparison
- Contingency
- Expansion
- Temporal

The tokenizer loads from `microsoft/deberta-v3-large`.

The model repository is hosted separately on Hugging Face. Given the repository may be private, authentication and permission may be required to access it before the running of `setup_models.py`.

## Running the Pipeline

### 1. Collecting Articles and Computing Features

Run:

```bash
python collect.py
```

First, the collection strategy is chosen under `collect.py`.

Following a successful processing of events, outputs are produced:

```text
metadata.csv
frozen_metadata.csv
framing_dataset.csv
proposal_labeled_features.csv
framing_rankings.csv
```

Extracted article text is stored with and connected to its related event data.

Discovered-event mode, in addition, carries out event-discovery and provides cluster-related outputs.

### 2. Framing Index Validation

After `framing_dataset.csv` files have been created, run:

```bash
python validate.py
```

This validation pipeline performs statistical analyses on the features of the Framing Index.

The current validation procedure includes:

- descriptive statistics,
- synthetic criterion-score construction,
- Pearson correlation,
- Spearman correlation,
- feature-correlation analysis,
- source-level summaries,
- Elastic Net regression,
- cross-validated performance measures,
- MAE, RMSE, and R² evaluation, and
- bootstrap coefficient stability analysis.

Validation outputs are stored in:

```text
multi_event_framing_project/validation_outputs/
```

The resulting outputs are:

```text
criterion_correlations.csv
source_summary.csv
feature_correlations.csv
elastic_net_predictions.csv
elastic_net_coefficients.csv
bootstrap_coefficients.csv
```

## Article Audit

The validation pipeline includes an article-level relevance and, by extension, data quality, check:

```text
multi_event_framing_project/validation_outputs/article_relevance_audit_1.csv
```

This file preserves human-coded relevance judgments.

If human-coded `relevance_label` values are available, the validation pipeline is able to restrict analysis to only those articles labeled as relevant. If there are no human-coded labels, validation is unfiltered.

Given this file contains human annotation, it should be stored for potential future use rather than be treated as an easily reproducible output.

## Validation Output

After running `validate.py`, generate index visualizations using:

```bash
python make_results_figures.py
```

The output includes:

```text
figure_1_framing_index_vs_synthetic_score.png
figure_2_elastic_net_coefficients.png
figure_3_bootstrap_coefficients.png
figure_4_source_median_index.png
figure_5_feature_correlation_matrix.png
```

The output is stored in:

```text
multi_event_framing_project/validation_outputs/
```

## Run Order

To run these files in a new environment, the sequence is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python setup_models.py
python collect.py
python validate.py
python make_results_figures.py
```

After setting up the new environment, analysis begins with `collect.py`.

## Outputs

At the event level, the project produces article text, feature scores, Framing Index values, and articles ranked by index values.

Towards validation, it produces correlation analyses, model coefficients, cross-validated performance measures, bootstrap estimates, Framing Index scores by source, and diagnostic values.

## Reproducibility

The repository is structured such that a user's computer-specific paths are not required for reproducing operations.

`requirements.txt` tracks python package versions, and `setup_models.py` downloads the PDTB/DeBERTa checkpoint.

Any human-generated annotations should be stored for future use.

## Research Status

This repository is an implementation of a Framing Index being actively developed. This will evolve as the methodology changes.

Results should be interpreted in the context of event corpora, features included, and validation approaches used in the analyses.