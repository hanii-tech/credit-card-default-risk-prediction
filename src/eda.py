"""Exploratory analysis for the credit-card default dataset.

Run from the project root with:

    python src/eda.py

The script prints summaries for the main business questions and saves plots
to ``data/eda_plots``.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_credit_card_data.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "figures"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset and add readable categorical group columns."""
    data = pd.read_csv(path)

    data["default"] = data["default"].astype(int)
    data["default_label"] = data["default"].map({0: "No default", 1: "Default"})

    education_columns = [
        column for column in data.columns if column.startswith("EDUCATION_")
    ]
    if education_columns:
        education_code = data[education_columns].astype(int).idxmax(axis=1)
        data["education_group"] = education_code.str.replace(
            "EDUCATION_", "", regex=False
        )
        no_education_dummy = data[education_columns].sum(axis=1).eq(0)
        data.loc[no_education_dummy, "education_group"] = "1"
        data["education_group"] = data["education_group"].map(
            {
                "1": "Graduate school",
                "2": "University",
                "3": "High school",
                "4": "Other",
            }
        )

    return data


def print_overview(data: pd.DataFrame) -> None:
    """Print counts, distributions, and default comparisons."""
    default_counts = data["default_label"].value_counts().reindex(
        ["No default", "Default"], fill_value=0
    )
    default_rates = data.groupby("default_label", observed=False)["default"].agg(
        customers="size", default_rate="mean"
    )

    print("Dataset shape:", data.shape)
    print("\nCustomers by outcome:")
    print(default_counts.to_string())
    print(f"\nTotal customers defaulted: {int(default_counts['Default'])}")
    print(f"Overall default rate: {data['default'].mean():.2%}")

    print("\nCredit limit by outcome:")
    print(data.groupby("default_label", observed=False)["LIMIT_BAL"].describe().round(2))

    print("\nAge by outcome:")
    print(data.groupby("default_label", observed=False)["AGE"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2))

    if "education_group" in data:
        education_summary = data.groupby("education_group", observed=False).agg(
            customers=("default", "size"),
            defaults=("default", "sum"),
            default_rate=("default", "mean"),
        )
        print("\nEducation and default:")
        print(education_summary.sort_values("default_rate", ascending=False).round(4))

    repayment_columns = [f"PAY_{period}" for period in [0, 2, 3, 4, 5, 6]]
    repayment_summary = data.groupby("default_label", observed=False)[repayment_columns].mean()
    data["severe_delinquency_count"] = (data[repayment_columns] >= 2).sum(axis=1)
    delinquency_summary = data.groupby("default_label", observed=False)[
        "severe_delinquency_count"
    ].mean()
    print("\nAverage repayment history by outcome:")
    print(repayment_summary.round(2).to_string())
    print("\nAverage number of months with serious delinquency (PAY >= 2):")
    print(delinquency_summary.round(2).to_string())

    bill_columns = [f"BILL_AMT{period}" for period in range(1, 7)]
    payment_columns = [f"PAY_AMT{period}" for period in range(1, 7)]
    data["average_bill_amount"] = data[bill_columns].mean(axis=1)
    data["average_payment_amount"] = data[payment_columns].mean(axis=1)
    amounts_summary = data.groupby("default_label", observed=False)[
        ["average_bill_amount", "average_payment_amount"]
    ].mean()
    print("\nAverage bill and payment amounts by outcome:")
    print(amounts_summary.round(2).to_string())


def print_feature_relationships(data: pd.DataFrame) -> None:
    """Rank numeric features by their absolute correlation with default."""
    numeric_data = data.select_dtypes(include="number").drop(
        columns=["default"], errors="ignore"
    )
    correlations = numeric_data.corrwith(data["default"]).dropna()
    relationships = pd.DataFrame(
        {"correlation": correlations, "absolute_correlation": correlations.abs()}
    ).sort_values("absolute_correlation", ascending=False)

    print("\nFeatures with the strongest linear relationship to default:")
    print(relationships.head(15).round(4).to_string())


def save_plots(data: pd.DataFrame, output_dir: Path = PLOT_DIR) -> None:
    """Save plots covering outcome, demographics, education, repayment, and amounts."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    default_counts = data["default_label"].value_counts().reindex(
        ["No default", "Default"], fill_value=0
    )
    default_percentages = default_counts / default_counts.sum() * 100
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(default_counts.index, default_counts.values, color=["#4c78a8", "#e45756"])
    axis.set_title("Credit-card default distribution")
    axis.set_ylabel("Number of customers")
    axis.set_ylim(0, default_counts.max() * 1.15)
    for bar, percentage in zip(bars, default_percentages):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "default_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    data.groupby("default_label", observed=False)["LIMIT_BAL"].plot(
        kind="hist", alpha=0.55, bins=30, legend=True
    )
    plt.xlabel("Credit limit")
    plt.title("Credit limit distribution by default outcome")
    plt.tight_layout()
    plt.savefig(output_dir / "credit_limit_by_default.png", dpi=150)
    plt.close()

    data.boxplot(column="AGE", by="default_label")
    plt.suptitle("")
    plt.title("Age by default outcome")
    plt.xlabel("")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(output_dir / "age_by_default.png", dpi=150)
    plt.close()

    if "education_group" in data:
        education_rates = data.groupby("education_group", observed=False)["default"].mean()
        education_rates.sort_values().plot(kind="bar", color="#d97706")
        plt.title("Default rate by education")
        plt.xlabel("")
        plt.ylabel("Default rate")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "education_default_rate.png", dpi=150)
        plt.close()

    repayment_columns = [f"PAY_{period}" for period in [0, 2, 3, 4, 5, 6]]
    repayment_means = data.groupby("default_label", observed=False)[repayment_columns].mean()
    repayment_means.T.plot(kind="bar", figsize=(9, 5))
    plt.title("Average repayment status by default outcome")
    plt.xlabel("Repayment month")
    plt.ylabel("Average status")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "repayment_history_by_default.png", dpi=150)
    plt.close()

    data["average_bill_amount"] = data[[f"BILL_AMT{period}" for period in range(1, 7)]].mean(axis=1)
    data["average_payment_amount"] = data[[f"PAY_AMT{period}" for period in range(1, 7)]].mean(axis=1)
    data.boxplot(
        column=["average_bill_amount", "average_payment_amount"],
        by="default_label",
        figsize=(9, 5),
        showfliers=False,
    )
    plt.suptitle("")
    plt.title("Average bills and payments by default outcome")
    plt.xlabel("")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(output_dir / "bills_payments_by_default.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


def main() -> None:
    data = load_data()
    print_overview(data)
    print_feature_relationships(data)
    save_plots(data)


if __name__ == "__main__":
    main()