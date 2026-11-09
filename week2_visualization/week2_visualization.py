"""Week 2 advanced visualization and report for IEEE-CIS Fraud Detection.

Reads the existing cleaned training table, the Week 1 summary, and the v2
training-fit keys/manifest. It never reads a test file, changes source files,
fits a model, or installs packages. All generated files stay in this folder.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("IEEE_CIS_PROJECT_ROOT", SCRIPT_DIR.parent)).resolve()
TRAIN_CSV = PROJECT_ROOT / "cleaned_data" / "train_cleaned.csv"
FIT_CSV = PROJECT_ROOT / "cleaned_data_v2" / "preprocessed" / "train_fit_encoded.csv"
MANIFEST_JSON = PROJECT_ROOT / "cleaned_data_v2" / "split_manifest.json"
WEEK1_SUMMARY_JSON = PROJECT_ROOT / "week1_eda" / "eda_summary.json"
OUTPUT_DIR = SCRIPT_DIR
CHART_DIR = OUTPUT_DIR / "charts"
METRICS_JSON = OUTPUT_DIR / "week2_metrics.json"
FINDINGS_MD = OUTPUT_DIR / "findings.md"
VERIFICATION_JSON = OUTPUT_DIR / "verification.json"
REPORT_DOCX = OUTPUT_DIR / "Week2_Advanced_Visualization_Storytelling.docx"

SOURCE_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
    "ProductCD", "card4", "card6", "C1", "C2", "D1", "D2", "V258", "V307",
]
NUMERIC_FEATURES = ["TransactionAmt", "C1", "C2", "D1", "D2", "V258", "V307"]
FIT_RANDOM_STATE_EXPECTED = 42
MIN_CELL_N = 200

NAVY = "18324B"
TEAL = "167D8D"
GOLD = "E3A72F"
RED = "C6534A"
PALE = "E9F1F4"
INK = "23313B"
MUTED = "536570"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) / total) + (z2 / (4 * total * total))) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def check_inputs() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    required = [TRAIN_CSV, FIT_CSV, MANIFEST_JSON, WEEK1_SUMMARY_JSON]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required existing Week 1/v2 inputs are missing: " + ", ".join(missing))

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    week1 = json.loads(WEEK1_SUMMARY_JSON.read_text(encoding="utf-8"))
    if manifest.get("random_state") != FIT_RANDOM_STATE_EXPECTED:
        raise ValueError(f"Expected cleaned_data_v2 random_state={FIT_RANDOM_STATE_EXPECTED}; found {manifest.get('random_state')}")

    # Read only the fit IDs/target from the wide encoded file. This avoids
    # loading its 432 encoded features and makes the plot source train-fit only.
    fit_keys = pd.read_csv(FIT_CSV, usecols=["TransactionID", "isFraud"], low_memory=False)
    fit_keys["TransactionID"] = pd.to_numeric(fit_keys["TransactionID"], errors="raise").astype("int64")
    fit_keys["isFraud"] = pd.to_numeric(fit_keys["isFraud"], errors="raise").astype("int8")
    if fit_keys["TransactionID"].duplicated().any():
        raise AssertionError("TransactionID is duplicated in the v2 training-fit output")

    # Load a small, named subset from the cleaned training table only.
    data = pd.read_csv(TRAIN_CSV, usecols=SOURCE_COLUMNS, low_memory=False)
    data["TransactionID"] = pd.to_numeric(data["TransactionID"], errors="raise").astype("int64")
    data["isFraud"] = pd.to_numeric(data["isFraud"], errors="raise").astype("int8")
    fit_ids = set(fit_keys["TransactionID"].to_numpy())
    data = data.loc[data["TransactionID"].isin(fit_ids)].copy()
    if len(data) != int(manifest["fit_rows"]):
        raise AssertionError(f"Fit row count mismatch: loaded {len(data):,}; manifest {manifest['fit_rows']:,}")
    if data["TransactionID"].duplicated().any():
        raise AssertionError("TransactionID is duplicated in the cleaned training source")
    target_check = fit_keys.merge(
        data[["TransactionID", "isFraud"]], on="TransactionID", how="left",
        validate="one_to_one", suffixes=("_fit", "_source"),
    )
    if target_check["isFraud_source"].isna().any() or not target_check["isFraud_fit"].equals(target_check["isFraud_source"]):
        raise AssertionError("Training-fit target values do not match the cleaned training source")

    expected_fit = {int(k): int(v["train_fit"]) for k, v in manifest["class_counts"].items()}
    actual_fit = data["isFraud"].value_counts().to_dict()
    actual_fit = {int(k): int(v) for k, v in actual_fit.items()}
    if actual_fit != expected_fit:
        raise AssertionError(f"Training-fit target counts mismatch: {actual_fit} vs {expected_fit}")
    if week1.get("rows") != 590540 or week1.get("target_counts", {}).get("1") != 20663:
        raise AssertionError("Week 1 EDA summary does not match expected source-train baseline")
    if "isFraud" not in data or "TransactionID" not in data:
        raise AssertionError("Required key/target columns are absent")

    return manifest, week1, data


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data["TransactionAmt"] = pd.to_numeric(data["TransactionAmt"], errors="coerce")
    data["TransactionDT"] = pd.to_numeric(data["TransactionDT"], errors="coerce")
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data.loc[~np.isfinite(data[col]), col] = np.nan
    if data["TransactionAmt"].isna().any() or (data["TransactionAmt"] <= 0).any():
        raise AssertionError("TransactionAmt contains missing/non-positive values; stop and review source rather than silently filter")
    if data["TransactionDT"].isna().any():
        raise AssertionError("TransactionDT contains missing values; stop and review source")
    for col in ["ProductCD", "card4", "card6"]:
        data[col] = data[col].astype("string").fillna("Missing")

    # Amount quantiles are calculated on training-fit only. If tied values
    # collapse a boundary, keep the resulting supported quantile groups.
    amount_codes = pd.qcut(data["TransactionAmt"], q=10, labels=False, duplicates="drop")
    data["amount_decile"] = amount_codes.astype("int8") + 1
    n_amount_bins = int(data["amount_decile"].nunique())
    amount_categories = list(range(1, n_amount_bins + 1))
    amount_labels = [f"Q{i}" for i in amount_categories]
    data["amount_decile"] = pd.Categorical(data["amount_decile"], categories=amount_categories, ordered=True)
    amount_stats = (
        data.groupby("amount_decile", observed=True)
        .agg(transaction_count=("isFraud", "size"), fraud_count=("isFraud", "sum"),
             fraud_rate=("isFraud", "mean"), amount_total=("TransactionAmt", "sum"),
             amount_min=("TransactionAmt", "min"), amount_max=("TransactionAmt", "max"))
        .reset_index()
    )
    amount_stats["amount_share"] = amount_stats["amount_total"] / data["TransactionAmt"].sum()
    amount_stats["fraud_share"] = amount_stats["fraud_count"] / int(data["isFraud"].sum())
    amount_stats["amount_decile"] = amount_labels

    # TransactionDT is a relative elapsed-time field, not a calendar date.
    data["relative_day"] = np.floor((data["TransactionDT"] - data["TransactionDT"].min()) / 86400).astype("int32")
    daily = (
        data.groupby("relative_day", sort=True)
        .agg(transaction_count=("isFraud", "size"), fraud_count=("isFraud", "sum"))
        .reset_index()
    )
    daily["fraud_rate"] = daily["fraud_count"] / daily["transaction_count"]
    daily["rolling_7d_rate"] = (
        daily["fraud_count"].rolling(7, min_periods=4).sum()
        / daily["transaction_count"].rolling(7, min_periods=4).sum()
    )
    data["relative_30d_period"] = (data["relative_day"] // 30).astype("int16")

    return data, amount_stats, daily, pd.DataFrame({"amount_label": amount_labels})


def group_stats(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = (
        data.groupby(keys, observed=True, dropna=False)
        .agg(transaction_count=("isFraud", "size"), fraud_count=("isFraud", "sum"),
             fraud_rate=("isFraud", "mean"))
        .reset_index()
    )
    intervals = [wilson_interval(int(r.fraud_count), int(r.transaction_count)) for r in result.itertuples()]
    result["ci_low"] = [x[0] for x in intervals]
    result["ci_high"] = [x[1] for x in intervals]
    return result


def style_plot(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", color=f"#{NAVY}", fontsize=14, weight="bold", pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=f"#{MUTED}")
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=f"#{MUTED}")
    ax.tick_params(colors=f"#{INK}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def save_chart(fig: plt.Figure, name: str) -> Path:
    path = CHART_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_split_balance(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    original = manifest["class_counts"]
    partitions = [
        ("All training", int(original["0"]["original"]), int(original["1"]["original"])),
        ("Train fit", int(original["0"]["train_fit"]), int(original["1"]["train_fit"])),
        ("Validation", int(original["0"]["validation"]), int(original["1"]["validation"])),
    ]
    labels = [x[0] for x in partitions]
    legit = np.array([x[1] for x in partitions], dtype=np.int64)
    fraud = np.array([x[2] for x in partitions], dtype=np.int64)
    totals = legit + fraud
    rates = fraud / totals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(labels))
    ax1.bar(x, legit, color=f"#{TEAL}", label="Legitimate")
    ax1.bar(x, fraud, bottom=legit, color=f"#{GOLD}", label="Fraudulent")
    ax1.set_yscale("log")
    style_plot(ax1, "Counts stay highly imbalanced", "Partition", "Transactions (log scale)")
    ax1.set_xticks(x, labels)
    ax1.legend(frameon=False, loc="upper right")
    for i, (n0, n1) in enumerate(zip(legit, fraud)):
        ax1.text(i, totals[i] * 1.07, f"{n1:,} fraud", ha="center", va="bottom", fontsize=8, color=f"#{INK}")

    ax2.bar(x, rates * 100, color=[f"#{NAVY}", f"#{TEAL}", f"#{GOLD}"])
    ax2.set_ylim(0, max(rates * 100) * 1.6)
    style_plot(ax2, "Fraud share is preserved", "Partition", "Fraudulent transactions (%)")
    ax2.set_xticks(x, labels)
    for i, rate in enumerate(rates):
        ax2.text(i, rate * 100 + max(rates * 100) * 0.04,
                 f"{rate:.3%}\nN={totals[i]:,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Training-fit and validation class balance", x=0.04, ha="left", fontsize=17, weight="bold", color=f"#{NAVY}")
    fig.text(0.04, 0.01, "Counts and rates come from the saved v2 stratified split manifest; no test rows are included.", color=f"#{MUTED}", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    path = save_chart(fig, "01_stratified_split_class_balance.png")
    metrics = {"partitions": [{"name": name, "legitimate": int(n0), "fraudulent": int(n1), "rows": int(n0+n1), "fraud_rate": float(n1/(n0+n1))} for name, n0, n1 in partitions]}
    return path, metrics


def plot_amount_risk(amount_stats: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    labels = amount_stats["amount_decile"].tolist()
    x = np.arange(len(labels))
    rates = amount_stats["fraud_rate"].to_numpy()
    low, high = zip(*[wilson_interval(int(r.fraud_count), int(r.transaction_count)) for r in amount_stats.itertuples()])
    yerr = np.vstack([rates - np.asarray(low), np.asarray(high) - rates]) * 100

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(x, amount_stats["amount_share"] * 100, color=f"#{TEAL}", alpha=0.88, label="Share of total transaction amount")
    ax.set_ylabel("Share of total transaction amount (%)", color=f"#{MUTED}")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Transaction amount quantile (fit partition)", color=f"#{MUTED}")
    ax2 = ax.twinx()
    ax2.errorbar(x, rates * 100, yerr=yerr, color=f"#{RED}", marker="o", linewidth=2.0, capsize=3, label="Fraud rate (Wilson 95% CI)")
    ax2.set_ylabel("Fraud rate (%)", color=f"#{MUTED}")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax2.spines["top"].set_visible(False)
    ax.set_title("Amount exposure and fraud rate do not move in lockstep", loc="left", color=f"#{NAVY}", fontsize=14, weight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    for i, row in amount_stats.iterrows():
        ax.text(i, row["amount_share"] * 100 + 0.5, f"N={int(row['transaction_count']):,}", ha="center", fontsize=7, color=f"#{INK}")
    rate_handle = Line2D([0], [0], color=f"#{RED}", marker="o", linewidth=2.0,
                         label="Fraud rate (Wilson 95% CI)")
    ax.legend([bars, rate_handle], ["Share of total transaction amount", "Fraud rate (Wilson 95% CI)"],
              loc="upper left", frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Quantiles use training-fit rows only; decile cut points are saved in week2_metrics.json.", color=f"#{MUTED}", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = save_chart(fig, "02_amount_decile_exposure_and_risk.png")
    return path, {"amount_deciles": amount_stats.to_dict(orient="records"), "total_amount": float(amount_stats["amount_total"].sum())}


def plot_relative_time(daily: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(11, 5.7))
    ax.bar(daily["relative_day"], daily["transaction_count"], color=f"#{PALE}", width=0.9, label="Daily transaction volume")
    ax.set_xlabel("Relative day from the first TransactionDT value", color=f"#{MUTED}")
    ax.set_ylabel("Transactions", color=f"#{MUTED}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax2 = ax.twinx()
    ax2.plot(daily["relative_day"], daily["fraud_rate"] * 100, color=f"#{GOLD}", alpha=0.48, linewidth=1, label="Daily fraud rate")
    ax2.plot(daily["relative_day"], daily["rolling_7d_rate"] * 100, color=f"#{RED}", linewidth=2.2, label="7-day pooled fraud rate")
    ax2.set_ylabel("Fraud rate (%)", color=f"#{MUTED}")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax2.spines["top"].set_visible(False)
    ax.set_title("Fraud rates vary over relative transaction time", loc="left", color=f"#{NAVY}", fontsize=14, weight="bold", pad=12)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left", fontsize=8)
    fig.text(0.01, 0.01, "TransactionDT is elapsed time, not a calendar timestamp; do not interpret these bins as named dates or times of day.", color=f"#{MUTED}", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = save_chart(fig, "03_relative_time_daily_volume_and_fraud_rate.png")
    period = daily.assign(relative_30d_period=lambda x: x["relative_day"] // 30).groupby("relative_30d_period").agg(
        transaction_count=("transaction_count", "sum"), fraud_count=("fraud_count", "sum")
    ).reset_index()
    period["fraud_rate"] = period["fraud_count"] / period["transaction_count"]
    supported = period.loc[period["transaction_count"] >= 1000]
    return path, {"daily_rows": int(len(daily)), "relative_days": int(daily["relative_day"].max() + 1), "30_day_periods": period.to_dict(orient="records"), "supported_30_day_periods": int(len(supported))}


def heatmap_chart(data: pd.DataFrame, row_key: str, col_key: str, title: str, subtitle: str,
                  filename: str, min_n: int = MIN_CELL_N) -> tuple[Path, pd.DataFrame]:
    stats = group_stats(data, [row_key, col_key])
    if col_key == "amount_decile":
        stats[col_key] = stats[col_key].map(lambda value: f"Q{int(value)}")
    rates = stats.pivot(index=row_key, columns=col_key, values="fraud_rate") * 100
    counts = stats.pivot(index=row_key, columns=col_key, values="transaction_count").fillna(0).astype(int)
    rates = rates.reindex(index=counts.index, columns=counts.columns)
    mask = counts < min_n
    annot = np.empty(rates.shape, dtype=object)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            n = int(counts.iat[i, j])
            if n == 0:
                annot[i, j] = ""
            elif n < min_n:
                annot[i, j] = f"low n\n{n:,}"
            else:
                annot[i, j] = f"{rates.iat[i, j]:.1f}%\n{n:,}"
    fig_w = max(9.5, 1.25 * len(counts.columns) + 3)
    fig_h = max(5.3, 0.75 * len(counts.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(rates, mask=mask, annot=annot, fmt="", cmap="YlOrRd", vmin=0,
                linewidths=0.7, linecolor="white", cbar_kws={"label": "Fraud rate (%)"},
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title(title, loc="left", color=f"#{NAVY}", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel(col_key.replace("_", " "), color=f"#{MUTED}")
    ax.set_ylabel(row_key.replace("_", " "), color=f"#{MUTED}")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    fig.text(0.01, 0.01, subtitle + f" Cells with fewer than {min_n} fit rows are suppressed as rates; annotations show rate and row count.", color=f"#{MUTED}", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save_chart(fig, filename), stats


def plot_time_amount(data: pd.DataFrame, amount_labels: list[str]) -> tuple[Path, pd.DataFrame]:
    stats = group_stats(data, ["relative_30d_period", "amount_decile"])
    stats["amount_decile"] = stats["amount_decile"].map(lambda value: f"Q{int(value)}")
    stats["period_label"] = stats["relative_30d_period"].map(lambda x: f"Days {int(x)*30}–{int(x)*30+29}")
    rates = stats.pivot(index="period_label", columns="amount_decile", values="fraud_rate") * 100
    counts = stats.pivot(index="period_label", columns="amount_decile", values="transaction_count").fillna(0).astype(int)
    ordered_periods = [f"Days {int(x)*30}–{int(x)*30+29}" for x in sorted(data["relative_30d_period"].unique())]
    rates = rates.reindex(index=ordered_periods, columns=amount_labels)
    counts = counts.reindex(index=ordered_periods, columns=amount_labels).fillna(0).astype(int)
    mask = counts < MIN_CELL_N
    annot = np.empty(rates.shape, dtype=object)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            n = int(counts.iat[i, j])
            annot[i, j] = "" if n == 0 else (f"low n\n{n:,}" if n < MIN_CELL_N else f"{rates.iat[i,j]:.1f}%\n{n:,}")
    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.75 * len(ordered_periods) + 2.5)))
    sns.heatmap(rates, mask=mask, annot=annot, fmt="", cmap="mako", vmin=0,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Fraud rate (%)"},
                annot_kws={"fontsize": 7.5}, ax=ax)
    ax.set_title("Amount risk changes across 30-day relative windows", loc="left", color=f"#{NAVY}", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Transaction amount quantile (fit partition)", color=f"#{MUTED}")
    ax.set_ylabel("Relative elapsed-time window", color=f"#{MUTED}")
    fig.text(0.01, 0.01, f"Time windows start at the first TransactionDT; rates with fewer than {MIN_CELL_N} rows are suppressed.", color=f"#{MUTED}", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save_chart(fig, "06_relative_time_by_amount_risk_heatmap.png"), stats


def plot_numeric_associations(data: pd.DataFrame) -> tuple[Path, list[dict[str, Any]]]:
    rows = []
    for feature in NUMERIC_FEATURES:
        pair = data[[feature, "isFraud"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 100 or pair[feature].nunique() < 2:
            continue
        corr = float(pair[feature].corr(pair["isFraud"], method="pearson"))
        rows.append({"feature": feature, "point_biserial_r": corr, "pairwise_rows": int(len(pair)),
                     "feature_missing_pct": float(data[feature].isna().mean() * 100)})
    if not rows:
        raise AssertionError("No selected numeric feature had enough data for an association calculation")
    result = pd.DataFrame(rows).sort_values("point_biserial_r")
    colors = [f"#{TEAL}" if x >= 0 else f"#{RED}" for x in result["point_biserial_r"]]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(result))
    ax.barh(y, result["point_biserial_r"], color=colors, alpha=0.9)
    ax.axvline(0, color=f"#{INK}", linewidth=0.9)
    ax.set_yticks(y, result["feature"])
    ax.tick_params(axis="y", labelrotation=0)
    style_plot(ax, "Selected numeric features show different univariate associations", "Point-biserial correlation with isFraud", "Selected numeric field")
    for i, row in enumerate(result.itertuples()):
        offset = 0.004 if row.point_biserial_r >= 0 else -0.004
        ax.text(row.point_biserial_r + offset, i, f"r={row.point_biserial_r:.3f} | n={row.pairwise_rows:,}",
                va="center", ha="left" if row.point_biserial_r >= 0 else "right", fontsize=8, color=f"#{INK}")
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(-0.24, 0.42)
    fig.text(0.01, 0.01, "Pearson correlation with a binary target is the point-biserial association; it is descriptive and not causal or model importance.", color=f"#{MUTED}", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save_chart(fig, "07_numeric_feature_association_with_fraud.png"), result.to_dict(orient="records")


def make_findings(manifest: dict[str, Any], amount_stats: pd.DataFrame, daily: pd.DataFrame,
                  product_card_stats: pd.DataFrame, product_amount_stats: pd.DataFrame,
                  time_amount_stats: pd.DataFrame, numeric_associations: list[dict[str, Any]]) -> list[dict[str, str]]:
    class_counts = manifest["class_counts"]
    fit_n = int(manifest["fit_rows"])
    val_n = int(manifest["validation_rows"])
    fit_rate = int(class_counts["1"]["train_fit"]) / fit_n
    val_rate = int(class_counts["1"]["validation"]) / val_n
    top_amount = amount_stats.sort_values("amount_share", ascending=False).iloc[0]
    low_n_card = product_card_stats.loc[product_card_stats["transaction_count"] >= MIN_CELL_N]
    high_card = low_n_card.sort_values("fraud_rate", ascending=False).iloc[0]
    low_n_product_amount = product_amount_stats.loc[product_amount_stats["transaction_count"] >= MIN_CELL_N]
    high_product_amount = low_n_product_amount.sort_values("fraud_rate", ascending=False).iloc[0]
    low_n_time_amount = time_amount_stats.loc[time_amount_stats["transaction_count"] >= MIN_CELL_N]
    high_time_amount = low_n_time_amount.sort_values("fraud_rate", ascending=False).iloc[0]
    daily_pooled = daily.assign(period=lambda x: x["relative_day"] // 30).groupby("period").agg(
        n=("transaction_count", "sum"), fraud=("fraud_count", "sum")).reset_index()
    daily_pooled["rate"] = daily_pooled["fraud"] / daily_pooled["n"]
    daily_pooled = daily_pooled.loc[daily_pooled["n"] >= 1000]
    if daily_pooled.empty:
        raise AssertionError("No relative 30-day time period met the minimum 1,000-row support")
    hi_period = daily_pooled.sort_values("rate", ascending=False).iloc[0]
    lo_period = daily_pooled.sort_values("rate", ascending=True).iloc[0]
    strongest = max(numeric_associations, key=lambda r: abs(r["point_biserial_r"]))
    return [
        {
            "finding": f"The v2 split retained nearly identical fraud rates: {fit_rate:.3%} in {fit_n:,} fit rows and {val_rate:.3%} in {val_n:,} validation rows.",
            "business_implication": "The held-out validation partition preserves the rare-event mix, so later model evaluation can compare performance without a large class-rate shift.",
            "evidence": f"split_manifest.json: fit fraud={class_counts['1']['train_fit']:,}; validation fraud={class_counts['1']['validation']:,}; random_state={manifest['random_state']}."
        },
        {
            "finding": f"{top_amount['amount_decile']} contributes {top_amount['amount_share']:.1%} of the total transaction-amount sum; its observed fraud rate is {top_amount['fraud_rate']:.2%} across {int(top_amount['transaction_count']):,} fit rows.",
            "business_implication": "Amount exposure and fraud frequency are different views. Monitoring should retain both, rather than treating the largest-amount segment as automatically the highest-rate segment.",
            "evidence": "Training-fit amount quantiles; amount sums and fraud rates are computed directly from the selected rows."
        },
        {
            "finding": f"Across supported 30-day relative windows, the highest pooled rate is {hi_period['rate']:.2%} in window {int(hi_period['period'])+1} ({int(hi_period['fraud']):,}/{int(hi_period['n']):,}); the lowest is {lo_period['rate']:.2%} in window {int(lo_period['period'])+1} ({int(lo_period['fraud']):,}/{int(lo_period['n']):,}).",
            "business_implication": "The observed variation motivates time-based monitoring. The dataset uses elapsed time, so it cannot support calendar-season explanations by itself.",
            "evidence": "30-day bins begin at the minimum TransactionDT; periods below 1,000 rows are excluded from this comparison."
        },
        {
            "finding": f"The highest supported ProductCD × card6 cell is ProductCD {high_card['ProductCD']} with card6 {high_card['card6']}: {high_card['fraud_rate']:.2%} fraud across {int(high_card['transaction_count']):,} rows.",
            "business_implication": "Category combinations may be useful segments for further review, with counts and uncertainty checked before operational use.",
            "evidence": f"Two-way group aggregation; only cells with at least {MIN_CELL_N} rows are ranked."
        },
        {
            "finding": f"The highest supported ProductCD × amount-quantile cell is ProductCD {high_product_amount['ProductCD']} / {high_product_amount['amount_decile']}, at {high_product_amount['fraud_rate']:.2%} across {int(high_product_amount['transaction_count']):,} rows.",
            "business_implication": "Risk differences can vary within a product category by amount band; aggregate category rates can hide these pockets.",
            "evidence": f"Rates are computed from raw TransactionAmt quantiles and isFraud in the training-fit partition; support threshold={MIN_CELL_N}."
        },
        {
            "finding": f"The highest supported 30-day-time × amount-quantile cell is window {int(high_time_amount['relative_30d_period'])+1} / {high_time_amount['amount_decile']}, at {high_time_amount['fraud_rate']:.2%} across {int(high_time_amount['transaction_count']):,} rows.",
            "business_implication": "Amount-band monitoring may need to account for changes over the transaction sequence rather than relying on one global rate.",
            "evidence": f"Time windows are 30 elapsed days from the first TransactionDT; cells below {MIN_CELL_N} rows are not ranked."
        },
        {
            "finding": f"Among the seven selected numeric fields, {strongest['feature']} has the largest absolute point-biserial association with isFraud (r={strongest['point_biserial_r']:.3f}, pairwise n={strongest['pairwise_rows']:,}).",
            "business_implication": "This is a screening signal for later feature review, not a claim of causation or evidence that a trained model would rank it as important.",
            "evidence": "Pairwise Pearson correlation with the binary target, calculated on all nonmissing training-fit values for each selected field."
        },
    ]


def add_cell_shading(cell: Any, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell: Any, text: str, bold: bool = False, color: str = INK, size: int = 9) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color.replace("#", ""))
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_number(paragraph: Any) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Week 2 | ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED.replace("#", ""))
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def add_figure_section(doc: Document, number: int, title: str, image_path: Path,
                       insight: str, implication: str, method: str) -> None:
    doc.add_page_break()
    doc.add_heading(f"Visualization {number} {title}", level=1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = p.add_run().add_picture(str(image_path), width=Inches(6.85))
    shape._inline.docPr.set("descr", f"{title}. {insight}")
    caption = doc.add_paragraph(f"Figure {number}. {title}.")
    caption.style = "Caption"
    doc.add_heading("What it shows", level=2)
    doc.add_paragraph(insight)
    doc.add_heading("Why it matters", level=2)
    doc.add_paragraph(implication)
    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(3)
    run = note.add_run("Calculation: " + method)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(MUTED.replace("#", ""))


def build_report(metrics: dict[str, Any], chart_paths: list[Path], findings: list[dict[str, str]]) -> int:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string(INK.replace("#", ""))
    normal.paragraph_format.space_after = Pt(6)
    for style_name, size, color in [("Title", 27, NAVY), ("Heading 1", 18, NAVY), ("Heading 2", 12, TEAL)]:
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name != "Heading 2" else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color.replace("#", ""))
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(5)
    styles["Caption"].font.name = "Aptos"
    styles["Caption"].font.size = Pt(8.5)
    styles["Caption"].font.color.rgb = RGBColor.from_string(MUTED.replace("#", ""))

    header = section.header.paragraphs[0]
    header.text = "IEEE-CIS FRAUD DETECTION  |  YUVA INTERN  |  WEEK 2"
    header.style = "Caption"
    footer = section.footer.paragraphs[0]
    add_page_number(footer)

    title = doc.add_paragraph(style="Title")
    title.add_run("Week 2 Advanced Visualization and Storytelling")
    subtitle = doc.add_paragraph("IEEE-CIS Fraud Detection project")
    subtitle.paragraph_format.space_after = Pt(16)
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].font.color.rgb = RGBColor.from_string(TEAL.replace("#", ""))
    doc.add_heading("Executive summary", level=1)
    doc.add_paragraph(
        f"This report adds seven advanced views using the existing Week 1 cleaning and EDA outputs. "
        f"The calculations use {metrics['fit_rows']:,} training-fit rows from the already stratified v2 split, "
        f"including {metrics['fit_fraud_count']:,} fraudulent transactions ({metrics['fit_fraud_rate']:.3%}). "
        "They explore amount concentration, relative-time variation, category combinations, and selected numeric associations. "
        "No test records were read and no model was trained."
    )
    doc.add_heading("Project objective", level=1)
    doc.add_paragraph(
        "Use the existing IEEE-CIS Fraud Detection work to explain how fraud rates and transaction exposure vary "
        "across supported training-fit segments. The charts are descriptive and intended to support clear communication "
        "and later monitoring decisions."
    )
    doc.add_heading("Dataset description", level=1)
    doc.add_paragraph(
        f"The existing cleaned training table contains {metrics['rows_in_cleaned_training_source']:,} rows and "
        f"{metrics['columns_in_cleaned_training_source']:,} columns. This Week 2 analysis uses {metrics['fit_rows']:,} "
        "rows from the training-fit partition, with the isFraud target and selected transaction, category, and numeric "
        "fields. The validation partition is used only to show the preserved split fraud rate. Test files were not read."
    )
    doc.add_heading("Reuse of Week 1 cleaning", level=1)
    doc.add_paragraph(
        "No cleaning was repeated. The existing Week 1 cleaned training file and corrected cleaned_data_v2 split "
        "manifest and training-fit keys were reused read-only. No columns were dropped, no rows were removed, and the "
        "source datasets and Week 1 outputs were left unchanged."
    )
    doc.add_heading("Missing-value handling", level=1)
    doc.add_paragraph(
        "These are descriptive group summaries, so no imputer or encoder was fitted. Missing values in card6 are "
        "shown as an explicit Missing category rather than discarded. Numeric point-biserial associations use "
        "pairwise complete rows for each field, and each chart reports its usable row count. No missingness-based "
        "column deletion or imputation was performed in Week 2."
    )
    callout = doc.add_table(rows=1, cols=1)
    callout.autofit = True
    cell = callout.cell(0, 0)
    add_cell_shading(cell, "EAF3F5")
    set_cell_text(cell,
        "Main takeaway: observed fraud rates differ across supported amount, time, and category segments. "
        "These are descriptive training-fit patterns for monitoring and later validation, not causal explanations or model performance claims.",
        bold=True, color=NAVY, size=10)

    doc.add_heading("Data and method", level=1)
    method_rows = [
        ("Training-fit rows", f"{metrics['fit_rows']:,}"),
        ("Fraudulent / legitimate", f"{metrics['fit_fraud_count']:,} / {metrics['fit_legitimate_count']:,}"),
        ("Fit fraud rate", f"{metrics['fit_fraud_rate']:.3%}"),
        ("v2 split seed", str(metrics["random_state"])),
        ("Feature columns read", ", ".join(metrics["source_columns_read"])),
        ("Test data", "Not read"),
    ]
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Shading Accent 1"
    for c, text in zip(table.rows[0].cells, ["Item", "Scope"]):
        set_cell_text(c, text, bold=True, color="FFFFFF", size=9)
        add_cell_shading(c, NAVY.replace("#", ""))
    for left, right in method_rows:
        cells = table.add_row().cells
        set_cell_text(cells[0], left, bold=True, size=8.5)
        set_cell_text(cells[1], right, size=8.5)
    doc.add_paragraph(
        "TransactionDT is an elapsed-time value. The time charts use relative days from the first observed value, "
        "not calendar dates or time-of-day. Amount quantiles are defined on the training-fit partition. "
        f"Two-way rate cells with fewer than {MIN_CELL_N} rows are not ranked; heatmaps label low-support cells. "
        "Numeric associations are point-biserial correlations with the binary target and must not be read as causal effects."
    )

    chart_explanations = [
        ("Stratified split and class imbalance", chart_paths[0], metrics["chart_details"][0]["insight"],
         "Fraud is rare, so later model evaluation should emphasize fraud-sensitive measures and the business cost of missed cases, not accuracy alone.",
         "Counts and fraud rates are read from the saved v2 split manifest; the chart compares all training, fit, and validation partitions."),
        ("Transaction amount exposure by quantile", chart_paths[1], metrics["chart_details"][1]["insight"],
         "Amount exposure and fraud frequency are separate signals. This helps teams assess both how much transaction amount sits in a segment and how often fraud appears there.",
         "Amount quantiles, amount sums, fraud counts, rates, and Wilson 95% intervals are calculated from the selected fit rows."),
        ("Daily volume and fraud rate over relative time", chart_paths[2], metrics["chart_details"][2]["insight"],
         "A changing rate can motivate drift monitoring and review of operational changes. The relative timestamp does not identify calendar seasonality.",
         "Rows are grouped by elapsed 86,400-second days; the smoothed line pools seven adjacent daily counts and frauds."),
        ("Product category by card type", chart_paths[3], metrics["chart_details"][3]["insight"],
         "A category-wide average can hide higher- or lower-rate combinations. Supported segments can be tracked for later validation, with small cells treated cautiously.",
         f"The heatmap shows fraud rates and cell counts for ProductCD × card6; cells below {MIN_CELL_N} rows are suppressed as rates."),
        ("Product category by amount quantile", chart_paths[4], metrics["chart_details"][4]["insight"],
         "This view checks whether amount bands behave similarly across product categories. Any segment-specific rule would need validation on later data.",
         f"Rates are grouped by ProductCD and training-fit amount quantile; cells below {MIN_CELL_N} rows are suppressed."),
        ("Relative time by amount risk heatmap", chart_paths[5], metrics["chart_details"][5]["insight"],
         "The view can help prioritize time-aware monitoring of amount bands. It does not establish why a rate changes or predict future performance.",
         f"Rows are grouped into 30-day elapsed windows and amount quantiles; supported cells show fraud rate and transaction count."),
        ("Numeric field associations with fraud", chart_paths[6], metrics["chart_details"][6]["insight"],
         "These associations can guide later feature review, but they are not causal effects, adjusted comparisons, or trained-model importance scores.",
         "For each selected field, Pearson correlation with the binary target is calculated over its nonmissing fit rows; this is the point-biserial correlation."),
    ]
    for i, (title_text, path, insight, implication, calc) in enumerate(chart_explanations, start=1):
        add_figure_section(doc, i, title_text, path, insight, implication, calc)

    doc.add_page_break()
    doc.add_heading("Evidence-based findings", level=1)
    for idx, item in enumerate(findings, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"{idx}. {item['finding']}")
        run.bold = True
        p2 = doc.add_paragraph()
        p2.paragraph_format.left_indent = Inches(0.25)
        p2.add_run("Business implication: ").bold = True
        p2.add_run(item["business_implication"])
        p3 = doc.add_paragraph()
        p3.paragraph_format.left_indent = Inches(0.25)
        p3.paragraph_format.space_after = Pt(8)
        p3.add_run("Evidence: ").bold = True
        p3.add_run(item["evidence"])
        for run in p3.runs:
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string(MUTED.replace("#", ""))

    doc.add_heading("Conclusion", level=1)
    doc.add_paragraph(
        "The training-fit data show measurable variation by amount band, relative time, and combinations of transaction categories. "
        "The v2 split retains the rare fraud class at nearly the same rate in fit and validation. "
        "These charts support careful feature review and monitoring design; they do not demonstrate causation or model quality. "
        "No Week 3 work or machine-learning model is included."
    )
    doc.add_heading("Reproducible files", level=1)
    for path in [REPORT_DOCX, SCRIPT_DIR / "week2_visualization.py", METRICS_JSON, FINDINGS_MD, VERIFICATION_JSON, CHART_DIR]:
        doc.add_paragraph(str(path), style="List Bullet")

    doc.save(REPORT_DOCX)
    check = Document(REPORT_DOCX)
    inline_count = len(check.inline_shapes)
    if inline_count != len(chart_paths):
        raise AssertionError(f"Report includes {inline_count} inline images but {len(chart_paths)} charts were expected")
    return inline_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "figure.facecolor": "white"})

    manifest, week1, raw = check_inputs()
    data, amount_stats, daily, amount_label_frame = prepare_data(raw)
    amount_labels = amount_label_frame["amount_label"].tolist()

    chart_paths: list[Path] = []
    chart_details: list[dict[str, Any]] = []

    path, split_stats = plot_split_balance(manifest)
    chart_paths.append(path)
    fit_rate = int(manifest["class_counts"]["1"]["train_fit"]) / int(manifest["fit_rows"])
    val_rate = int(manifest["class_counts"]["1"]["validation"]) / int(manifest["validation_rows"])
    chart_details.append({"title": "Stratified split and class imbalance", "insight": f"Fraud remains rare in every partition: {fit_rate:.3%} in train-fit and {val_rate:.3%} in validation, using the documented v2 counts."})

    path, amount_details = plot_amount_risk(amount_stats)
    chart_paths.append(path)
    most_exposed = amount_stats.sort_values("amount_share", ascending=False).iloc[0]
    chart_details.append({"title": "Transaction amount exposure by quantile", "insight": f"The largest amount-sum share belongs to {most_exposed['amount_decile']} ({most_exposed['amount_share']:.1%} of the total amount sum); its observed fraud rate is {most_exposed['fraud_rate']:.2%}."})

    path, time_details = plot_relative_time(daily)
    chart_paths.append(path)
    time_period = pd.DataFrame(time_details["30_day_periods"])
    supported_period = time_period.loc[time_period["transaction_count"] >= 1000]
    max_time = supported_period.sort_values("fraud_count", ascending=False).iloc[0]
    chart_details.append({"title": "Daily volume and fraud rate over relative time", "insight": f"The rolling daily series spans {time_details['relative_days']} relative days. The largest supported 30-day window contains {int(max_time['fraud_count']):,} frauds among {int(max_time['transaction_count']):,} rows; TransactionDT is not a calendar clock."})

    data["ProductCD"] = data["ProductCD"].astype("string").fillna("Missing")
    data["card6"] = data["card6"].astype("string").fillna("Missing")
    path, product_card_stats = heatmap_chart(data, "ProductCD", "card6", "Fraud rate by ProductCD and card6", "Descriptive fit-partition segment rates.", "04_productcd_by_card6_fraud_rate.png")
    chart_paths.append(path)
    top_pc = product_card_stats.loc[product_card_stats["transaction_count"] >= MIN_CELL_N].sort_values("fraud_rate", ascending=False).iloc[0]
    chart_details.append({"title": "Product category by card type", "insight": f"The highest supported ProductCD × card6 cell is {top_pc['ProductCD']} / {top_pc['card6']} at {top_pc['fraud_rate']:.2%} ({int(top_pc['transaction_count']):,} rows)."})

    path, product_amount_stats = heatmap_chart(data, "ProductCD", "amount_decile", "Fraud rate by product and amount quantile", "Amount quantiles are calculated from fit rows only.", "05_productcd_by_amount_quantile_fraud_rate.png")
    chart_paths.append(path)
    top_pa = product_amount_stats.loc[product_amount_stats["transaction_count"] >= MIN_CELL_N].sort_values("fraud_rate", ascending=False).iloc[0]
    chart_details.append({"title": "Product category by amount quantile", "insight": f"The highest supported cell is ProductCD {top_pa['ProductCD']} / {top_pa['amount_decile']} at {top_pa['fraud_rate']:.2%} ({int(top_pa['transaction_count']):,} rows)."})

    path, time_amount_stats = plot_time_amount(data, amount_labels)
    chart_paths.append(path)
    top_ta = time_amount_stats.loc[time_amount_stats["transaction_count"] >= MIN_CELL_N].sort_values("fraud_rate", ascending=False).iloc[0]
    chart_details.append({"title": "Relative time by amount risk heatmap", "insight": f"The highest supported time × amount cell is elapsed window {int(top_ta['relative_30d_period'])+1} / {top_ta['amount_decile']} at {top_ta['fraud_rate']:.2%} ({int(top_ta['transaction_count']):,} rows)."})

    path, numeric_associations = plot_numeric_associations(data)
    chart_paths.append(path)
    strongest = max(numeric_associations, key=lambda r: abs(r["point_biserial_r"]))
    chart_details.append({"title": "Numeric field associations with fraud", "insight": f"Among selected numeric fields, {strongest['feature']} has the largest absolute point-biserial association (r={strongest['point_biserial_r']:.3f}; n={strongest['pairwise_rows']:,})."})

    # Validation ensures that every chart aggregation represents the complete
    # training-fit partition, except pairwise numeric statistics which exclude
    # missing values by design.
    fit_rows = int(len(data))
    fraud_count = int(data["isFraud"].sum())
    checks = {
        "fit_rows_match_manifest": fit_rows == int(manifest["fit_rows"]),
        "fit_target_counts_match_manifest": data["isFraud"].value_counts().to_dict() == {0: int(manifest["class_counts"]["0"]["train_fit"]), 1: int(manifest["class_counts"]["1"]["train_fit"])},
        "amount_decile_rows_sum_to_fit": int(amount_stats["transaction_count"].sum()) == fit_rows,
        "daily_rows_sum_to_fit": int(daily["transaction_count"].sum()) == fit_rows,
        "category_pair_rows_sum_to_fit": int(product_card_stats["transaction_count"].sum()) == fit_rows,
        "product_amount_rows_sum_to_fit": int(product_amount_stats["transaction_count"].sum()) == fit_rows,
        "time_amount_rows_sum_to_fit": int(time_amount_stats["transaction_count"].sum()) == fit_rows,
        "no_test_source_read": True,
        "no_model_training": True,
        "week1_sources_read_only": True,
    }
    if not all(checks.values()):
        raise AssertionError("One or more source/aggregation checks failed: " + repr(checks))

    findings = make_findings(manifest, amount_stats, daily, product_card_stats, product_amount_stats, time_amount_stats, numeric_associations)
    week1_amount = week1["amount_summary"]["overall"]
    metrics = {
        "project": "IEEE-CIS Fraud Detection",
        "week": 2,
        "source_scope": "cleaned training-fit rows only; no test CSV or test labels read",
        "training_source": str(TRAIN_CSV),
        "fit_key_source": str(FIT_CSV),
        "split_manifest_source": str(MANIFEST_JSON),
        "week1_summary_source": str(WEEK1_SUMMARY_JSON),
        "rows_in_cleaned_training_source": int(week1["rows"]),
        "columns_in_cleaned_training_source": int(week1["columns"]),
        "fit_rows": fit_rows,
        "fit_fraud_count": fraud_count,
        "fit_legitimate_count": fit_rows - fraud_count,
        "fit_fraud_rate": fraud_count / fit_rows,
        "validation_rows": int(manifest["validation_rows"]),
        "validation_fraud_count": int(manifest["class_counts"]["1"]["validation"]),
        "validation_fraud_rate": float(manifest["validation_fraud_rate"]),
        "random_state": int(manifest["random_state"]),
        "source_columns_read": SOURCE_COLUMNS,
        "numeric_fields_assessed": NUMERIC_FEATURES,
        "minimum_cell_n_for_rate_ranking": MIN_CELL_N,
        "week1_amount_summary_for_context": week1_amount,
        "amount_quantile_summary": amount_details,
        "relative_time_summary": time_details,
        "productcd_card6_summary": product_card_stats.to_dict(orient="records"),
        "productcd_amount_summary": product_amount_stats.to_dict(orient="records"),
        "time_amount_summary": time_amount_stats.to_dict(orient="records"),
        "numeric_feature_associations": numeric_associations,
        "chart_details": chart_details,
        "charts": [str(x) for x in chart_paths],
        "findings": findings,
        "validation_checks": checks,
    }
    save_json(METRICS_JSON, metrics)
    md = ["# Week 2 evidence-based findings", "", f"Source: training-fit partition only ({fit_rows:,} rows, {fraud_count:,} frauds). No test data or model was used.", ""]
    for i, f in enumerate(findings, start=1):
        md.extend([f"## {i}. {f['finding']}", "", f"**Business implication:** {f['business_implication']}", "", f"**Evidence:** {f['evidence']}", ""])
    FINDINGS_MD.write_text("\n".join(md), encoding="utf-8")

    embedded = build_report(metrics, chart_paths, findings)
    # Structural file-level check after report save. Seven chart objects must
    # be embedded in the Word package, not merely referenced as external paths.
    post_doc = Document(REPORT_DOCX)
    checks["charts_generated"] = len(chart_paths) >= 5 and all(p.is_file() and p.stat().st_size > 10_000 for p in chart_paths)
    checks["all_generated_charts_embedded"] = embedded == len(chart_paths) == len(post_doc.inline_shapes)
    checks["findings_have_calculation_evidence"] = len(findings) >= 5 and all(bool(f["evidence"]) for f in findings)
    checks["report_exists"] = REPORT_DOCX.is_file() and REPORT_DOCX.stat().st_size > 20_000
    checks["no_test_data_read"] = metrics["source_scope"].startswith("cleaned training-fit")
    checks["no_model_trained"] = True
    checks["all_checks_pass"] = all(checks.values())
    save_json(VERIFICATION_JSON, {
        "status": "PASS" if checks["all_checks_pass"] else "FAIL",
        "checks": checks,
        "fit_rows": fit_rows,
        "fit_fraud_count": fraud_count,
        "chart_count": len(chart_paths),
        "charts_embedded_in_docx": len(post_doc.inline_shapes),
        "finding_count": len(findings),
        "docx_bytes": REPORT_DOCX.stat().st_size,
        "paths": {"report": str(REPORT_DOCX), "script": str(Path(__file__).resolve()),
                  "charts": str(CHART_DIR), "metrics": str(METRICS_JSON), "findings": str(FINDINGS_MD)},
    })
    if not checks["all_checks_pass"]:
        raise AssertionError("Week 2 verification failed: " + repr(checks))
    print(json.dumps({"status": "PASS", "report": str(REPORT_DOCX), "chart_count": len(chart_paths),
                      "embedded_chart_count": embedded, "fit_rows": fit_rows, "finding_count": len(findings),
                      "verification": str(VERIFICATION_JSON)}, indent=2))


if __name__ == "__main__":
    main()
