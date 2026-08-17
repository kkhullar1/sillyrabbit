from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = (
    Path("/Users/keshkhullar/PycharmProjects/PythonProject4")
    / "multi_event_framing_project"
    / "validation_outputs"
)

PREDICTIONS_FILE = OUTPUT_DIR / "elastic_net_predictions.csv"
COEFFICIENTS_FILE = OUTPUT_DIR / "elastic_net_coefficients.csv"
BOOTSTRAP_FILE = OUTPUT_DIR / "bootstrap_coefficients.csv"
SOURCE_SUMMARY_FILE = OUTPUT_DIR / "source_summary.csv"
FEATURE_CORR_FILE = OUTPUT_DIR / "feature_correlations.csv"


def save_scatterplot():
    df = pd.read_csv(PREDICTIONS_FILE)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["framing_index_consensus_adjusted"],
        df["synthetic_human_score_1_to_5"]
    )

    plt.xlabel("Framing Index")
    plt.ylabel("Synthetic Human Score")
    plt.title("Framing Index vs Synthetic Human Score")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure_1_framing_index_vs_synthetic_score.png",
        dpi=300
    )

    plt.close()


def save_elastic_net_bar_chart():
    df = pd.read_csv(COEFFICIENTS_FILE)

    df = df.sort_values(
        "abs_coefficient",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        df["feature"],
        df["coefficient"]
    )

    plt.xlabel("Elastic Net Coefficient")
    plt.title("Elastic Net Feature Coefficients")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure_2_elastic_net_coefficients.png",
        dpi=300
    )

    plt.close()


def save_bootstrap_ci_chart():
    df = pd.read_csv(BOOTSTRAP_FILE)

    df = df.sort_values(
        "abs_mean",
        ascending=True
    )

    lower_error = (
        df["mean_coefficient"] - df["lower_95"]
    )

    upper_error = (
        df["upper_95"] - df["mean_coefficient"]
    )

    plt.figure(figsize=(10, 7))

    plt.errorbar(
        df["mean_coefficient"],
        df["feature"],
        xerr=[
            lower_error,
            upper_error
        ],
        fmt="o"
    )

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.xlabel("Bootstrap Mean Coefficient")
    plt.title("Bootstrap Coefficient Stability with 95% Intervals")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure_3_bootstrap_coefficients.png",
        dpi=300
    )

    plt.close()


def save_source_summary_chart():
    df = pd.read_csv(SOURCE_SUMMARY_FILE)

    df = df.sort_values(
        "median_index",
        ascending=True
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        df["source"],
        df["median_index"]
    )

    plt.xlabel("Median Framing Index")
    plt.title("Median Framing Index by Source")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure_4_source_median_index.png",
        dpi=300
    )

    plt.close()


def save_feature_correlation_heatmap():
    df = pd.read_csv(
        FEATURE_CORR_FILE,
        index_col=0
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(
        df,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(df.columns)),
        df.columns,
        rotation=90
    )

    plt.yticks(
        range(len(df.index)),
        df.index
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure_5_feature_correlation_matrix.png",
        dpi=300
    )

    plt.close()


if __name__ == "__main__":

    save_scatterplot()
    save_elastic_net_bar_chart()
    save_bootstrap_ci_chart()
    save_source_summary_chart()
    save_feature_correlation_heatmap()

    print("Saved figures to:")
    print(OUTPUT_DIR)