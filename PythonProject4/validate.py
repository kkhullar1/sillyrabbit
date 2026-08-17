import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("multi_event_framing_project")

EVENT_DIRS = [
    Path("rss_a_plus_corpus"),
    Path("event_us_inflation"),
    Path("event_israel_iran"),
    Path("event_ai_regulation"),
    Path("event_fed_rates"),
    Path("event_nato_summit"),
    Path("event_climate_policy"),
    Path("event_china_tariffs"),
    Path("event_ukraine_war"),
    Path("event_world_cup"),
    Path("event_trump_administration"),
    Path("event_russia_ukraine"),
    Path("event_spacex"),
    Path("event_extreme_weather"),
    Path("event_israel_gaza"),
    Path("event_israel_lebanon"),
    Path("event_climate_change"),
    Path("event_heat_wave"),
    Path("event_anthropic"),
    Path("event_us_china_trade"),
    Path("event_federal_budget")
]

OUTPUT_DIR = BASE_DIR / "validation_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "hedge_rate_deviation",
    "emotional_amplification_deviation",
    "tfidf_discourse_emphasis_deviation",
    "omission_score",
    "content_deviation_deviation",
    "stance_misalignment_deviation",
    "causal_coherence_deviation"
]


def clean_event_name(path):
    name = path.name
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


def load_event_dataset(event_dir):
    dataset_path = event_dir / "framing_dataset.csv"

    if not dataset_path.exists():
        print(f"Missing framing_dataset.csv in {event_dir}")
        return pd.DataFrame()

    df = pd.read_csv(dataset_path)
    df["event"] = clean_event_name(event_dir)

    return df


def load_all_events(event_dirs):
    frames = []

    for event_dir in event_dirs:
        df = load_event_dataset(event_dir)

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def minmax_scale(series):
    series = pd.to_numeric(series, errors="coerce")

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - minimum) / (maximum - minimum)


def require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

def fill_missing_feature_values(df):
    df = df.copy()

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[column] = df[column].fillna(median_value)

    return df


def synthesize_human_scores(df):
    df = df.copy()

    needed_columns = [
        "emotional_amplification_deviation",
        "stance_misalignment_deviation",
        "content_deviation_deviation",
        "tfidf_discourse_emphasis_deviation",
        "omission_score",
        "causal_coherence_deviation"
    ]

    require_columns(df, needed_columns)

    df = fill_missing_feature_values(df)

    df["synthetic_human_score_raw"] = (
        0.20 * minmax_scale(df["emotional_amplification_deviation"])
        + 0.20 * minmax_scale(df["stance_misalignment_deviation"])
        + 0.20 * minmax_scale(df["content_deviation_deviation"])
        + 0.15 * minmax_scale(df["tfidf_discourse_emphasis_deviation"])
        + 0.15 * minmax_scale(df["omission_score"])
        + 0.10 * minmax_scale(df["causal_coherence_deviation"])
    )

    df["synthetic_human_score_1_to_5"] = (
        1 + 4 * minmax_scale(df["synthetic_human_score_raw"])
    )

    return df

def create_article_key(df):
    return (
        df["event"].astype(str).str.strip().str.lower()
        + "||"
        + df["source"].astype(str).str.strip().str.lower()
        + "||"
        + df["title"].astype(str).str.strip().str.lower()
    )


def create_relevance_audit_file(master_df):
    audit_path = OUTPUT_DIR / "article_relevance_audit_1.csv"

    audit_columns = [
        "event",
        "source",
        "title",
        "url",
        "framing_index_consensus_adjusted",
        "synthetic_human_score_1_to_5"
    ]

    existing_columns = [
        column for column in audit_columns
        if column in master_df.columns
    ]

    new_audit_df = master_df[existing_columns].copy()
    new_audit_df["article_key"] = create_article_key(new_audit_df)

    if not audit_path.exists():
        new_audit_df["relevance_label"] = ""
        new_audit_df["relevance_notes"] = ""
        new_audit_df.to_csv(audit_path, index=False)

        print("\nSaved new article relevance audit file:")
        print(audit_path)

        return new_audit_df

    old_audit_df = pd.read_csv(audit_path)

    if "article_key" not in old_audit_df.columns:
        old_audit_df["article_key"] = create_article_key(old_audit_df)

    if "relevance_label" not in old_audit_df.columns:
        old_audit_df["relevance_label"] = ""

    if "relevance_notes" not in old_audit_df.columns:
        old_audit_df["relevance_notes"] = ""

    old_keys = set(old_audit_df["article_key"].astype(str).tolist())

    rows_to_add = new_audit_df[
        ~new_audit_df["article_key"].isin(old_keys)
    ].copy()

    rows_to_add["relevance_label"] = ""
    rows_to_add["relevance_notes"] = ""

    combined_audit_df = pd.concat(
        [old_audit_df, rows_to_add],
        ignore_index=True
    )

    combined_audit_df.to_csv(audit_path, index=False)

    print("\nExisting audit file updated:")
    print(audit_path)
    print("Existing rows:", len(old_audit_df))
    print("New rows added:", len(rows_to_add))
    print("Total audit rows:", len(combined_audit_df))

    return combined_audit_df

def apply_relevance_filter(master_df):
    audit_path = OUTPUT_DIR / "article_relevance_audit_1.csv"

    if not audit_path.exists():
        print("\nNo relevance audit file found.")
        print("Using unfiltered dataset.")
        return master_df

    audit_df = pd.read_csv(audit_path)

    if audit_df.empty:
        print("\nAudit file is empty.")
        print("Using unfiltered dataset.")
        return master_df

    if "relevance_label" not in audit_df.columns:
        print("\nNo relevance_label column found.")
        print("Using unfiltered dataset.")
        return master_df

    audit_df["relevance_label"] = pd.to_numeric(
        audit_df["relevance_label"],
        errors="coerce"
    )

    print("\nRelevance label counts:")
    print(audit_df["relevance_label"].value_counts(dropna=False))

    if audit_df["relevance_label"].notna().sum() == 0:
        print("\nNo manual relevance labels found yet.")
        print("Using unfiltered dataset.")
        return master_df

    audit_df = audit_df[audit_df["relevance_label"] == 1].copy()

    print("\nRows marked relevant:", len(audit_df))

    master_df = master_df.copy()
    master_df["article_key"] = create_article_key(master_df)

    before_count = len(master_df)

    master_df = (
        master_df
        .merge(
            audit_df[["article_key"]].drop_duplicates(),
            on="article_key",
            how="inner"
        )
    )

    master_df = master_df.drop(columns=["article_key"])

    print("\nRows before filtering:", before_count)
    print("Rows after filtering:", len(master_df))

    return master_df


def run_correlations(df):
    require_columns(
        df,
        [
            "framing_index_consensus_adjusted",
            "synthetic_human_score_1_to_5"
        ]
    )

    corr_df = df[
        [
            "framing_index_consensus_adjusted",
            "synthetic_human_score_1_to_5"
        ]
    ].copy()

    corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
    corr_df = corr_df.dropna()

    n = len(corr_df)

    if n < 10:
        return pd.DataFrame([
            {
                "comparison": "synthetic_human_vs_current_index",
                "n": n,
                "pearson_corr": np.nan,
                "pearson_p": np.nan,
                "spearman_corr": np.nan,
                "spearman_p": np.nan,
                "note": "Need at least 10 valid non-NaN rows"
            }
        ])

    pearson_corr, pearson_p = pearsonr(
        corr_df["synthetic_human_score_1_to_5"],
        corr_df["framing_index_consensus_adjusted"]
    )

    spearman_corr, spearman_p = spearmanr(
        corr_df["synthetic_human_score_1_to_5"],
        corr_df["framing_index_consensus_adjusted"]
    )

    return pd.DataFrame([
        {
            "comparison": "synthetic_human_vs_current_index",
            "n": n,
            "pearson_corr": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "note": "Synthetic target only"
        }
    ])

def run_elastic_net(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = fill_missing_feature_values(df)

    df = df.dropna(
        subset=["synthetic_human_score_1_to_5"]
    ).copy()

    print(
        "Rows available for Elastic Net after feature imputation:",
        len(df)
    )

    if len(df) < 15:
        print("Too few clean rows for Elastic Net.")
        return pd.DataFrame(), pd.DataFrame()

    X = df[FEATURE_COLUMNS]
    y = df["synthetic_human_score_1_to_5"]

    cv_splits = min(5, len(df))

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=np.logspace(-4, 1, 50),
                    cv=cv_splits,
                    max_iter=10000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X, y)

    predictions = model.predict(X)

    learned_df = df.copy()
    learned_df["elastic_net_prediction"] = predictions

    elastic_net = model.named_steps["elastic_net"]

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": elastic_net.coef_
        }
    )

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    coef_df = coef_df.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print()
    print("Cross-validation")

    n_splits = min(10, len(df))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    cv_predictions = cross_val_predict(
        model,
        X,
        y,
        cv=kf
    )

    print("R²:", round(r2_score(y, cv_predictions), 3))
    print("MAE:", round(mean_absolute_error(y, cv_predictions), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y, cv_predictions)), 3))

    return learned_df, coef_df

def bootstrap_coefficients(df, n_bootstraps=1000):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = fill_missing_feature_values(df)

    df = df.dropna(
        subset=["synthetic_human_score_1_to_5"]
    ).copy()

    if len(df) < 15:
        print("Too few rows for bootstrap stability.")
        return pd.DataFrame()

    X = df[FEATURE_COLUMNS]
    y = df["synthetic_human_score_1_to_5"]

    coefficients = []

    cv_splits = min(5, len(df))

    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(
            len(df),
            size=len(df),
            replace=True
        )

        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elastic_net",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.logspace(-4, 1, 50),
                        cv=cv_splits,
                        max_iter=10000,
                        random_state=42
                    )
                )
            ]
        )

        model.fit(X_sample, y_sample)

        elastic_net = model.named_steps["elastic_net"]

        coefficients.append(elastic_net.coef_)

    coefficients = np.array(coefficients)

    bootstrap_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_coefficient": coefficients.mean(axis=0),
            "std_coefficient": coefficients.std(axis=0),
            "lower_95": np.percentile(coefficients, 2.5, axis=0),
            "upper_95": np.percentile(coefficients, 97.5, axis=0)
        }
    )

    bootstrap_df["abs_mean"] = bootstrap_df["mean_coefficient"].abs()

    bootstrap_df = (
        bootstrap_df
        .sort_values("abs_mean", ascending=False)
        .reset_index(drop=True)
    )

    return bootstrap_df

if __name__ == "__main__":
    print("\nLoading event datasets...")

    master_df = load_all_events(EVENT_DIRS)

    print("Loaded rows:", len(master_df))

    if master_df.empty:
        raise SystemExit("No event datasets found.")

    print("\nRows per event:")
    print(master_df["event"].value_counts().sort_index())

    print("\nSynthesizing human scores...")
    master_df = synthesize_human_scores(master_df)

    create_relevance_audit_file(master_df)

    master_df = apply_relevance_filter(master_df)

    print("\nRunning criterion-style correlations...")
    correlation_df = run_correlations(master_df)
    print(correlation_df)

    correlation_df.to_csv(
        OUTPUT_DIR / "criterion_correlations.csv",
        index=False
    )

    print("\nSummarizing by source...")

    source_summary = (
        master_df
        .groupby("source")
        .agg(
            article_count=("source", "size"),
            mean_index=("framing_index_consensus_adjusted", "mean"),
            median_index=("framing_index_consensus_adjusted", "median"),
            mean_synthetic_score=("synthetic_human_score_1_to_5", "mean")
        )
        .reset_index()
        .sort_values("mean_index", ascending=False)
    )

    print(source_summary)

    source_summary.to_csv(
        OUTPUT_DIR / "source_summary.csv",
        index=False
    )

    print("\nComputing feature correlation matrix...")

    master_df = fill_missing_feature_values(master_df)

    feature_corr_df = (
        master_df[
            FEATURE_COLUMNS
            + [
                "framing_index_consensus_adjusted",
                "synthetic_human_score_1_to_5"
            ]
        ]
        .corr()
    )

    print(feature_corr_df)

    feature_corr_df.to_csv(
        OUTPUT_DIR / "feature_correlations.csv"
    )

    print("\nRunning Elastic Net...")

    learned_df, coef_df = run_elastic_net(master_df)

    print()
    print("Elastic Net coefficients:")
    print(coef_df)

    learned_df.to_csv(
        OUTPUT_DIR / "elastic_net_predictions.csv",
        index=False
    )

    coef_df.to_csv(
        OUTPUT_DIR / "elastic_net_coefficients.csv",
        index=False
    )

    print()
    print("Running bootstrap stability...")

    bootstrap_df = bootstrap_coefficients(
        master_df,
        n_bootstraps=1000
    )

    print()
    print(bootstrap_df)

    bootstrap_df.to_csv(
        OUTPUT_DIR / "bootstrap_coefficients.csv",
        index=False
    )

    print()
    print("Saved validation outputs to:")
    print(OUTPUT_DIR)
