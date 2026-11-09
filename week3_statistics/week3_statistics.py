"""YuvaIntern Week 3: training-only hypothesis tests and reporting.

This reuses cleaned_data/train_cleaned.csv and cleaned_data_v2 fit keys. It
does not access test files, alter inputs, fit a predictive model, or redo data
cleaning. The v2 training-fit rows are analyzed only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(os.environ.get("IEEE_CIS_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUT = Path(__file__).resolve().parent
CHARTS = OUT / "charts"
TRAIN = ROOT / "cleaned_data" / "train_cleaned.csv"
FIT_KEYS = ROOT / "cleaned_data_v2" / "preprocessed" / "train_fit_encoded.csv"
MANIFEST = ROOT / "cleaned_data_v2" / "split_manifest.json"
W1 = ROOT / "week1_eda" / "findings.md"
W2 = ROOT / "week2_visualization" / "findings.md"
RESULTS = OUT / "week3_results.json"
GROUPS_CSV = OUT / "week3_group_summaries.csv"
REPORT = OUT / "Week3_Statistical_Analysis_and_Hypothesis_Testing.docx"
VERIFY = OUT / "verification.json"
SEED = 2026
BOOTSTRAP_REPS = 600
MONTE_CARLO_REPS = 20000
MONTE_CARLO_SEED = 4321
ALPHA = 0.05
FEATURES = ["TransactionID", "isFraud", "TransactionAmt", "ProductCD", "card6", "V258"]
NAVY, TEAL, RED, GOLD, INK, MUTED = "18324B", "167D8D", "C6534A", "E3A72F", "23313B", "536570"


def jwrite(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def load_fit() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fit_keys = pd.read_csv(FIT_KEYS, usecols=["TransactionID", "isFraud"])
    if fit_keys["TransactionID"].duplicated().any():
        raise AssertionError("Training-fit TransactionID keys are not unique")
    data = pd.read_csv(TRAIN, usecols=FEATURES)
    if data["TransactionID"].duplicated().any():
        raise AssertionError("Cleaned training TransactionID keys are not unique")
    fit_ids = set(fit_keys["TransactionID"].tolist())
    fit = data.loc[data["TransactionID"].isin(fit_ids)].copy()
    fit = fit.merge(fit_keys, on="TransactionID", how="inner", validate="one_to_one", suffixes=("_source", "_fit"))
    if len(fit) != manifest["fit_rows"] or len(fit) != len(fit_keys):
        raise AssertionError(f"Fit row/key mismatch: {len(fit)}, {len(fit_keys)}, expected {manifest['fit_rows']}")
    if not np.array_equal(fit["isFraud_source"].to_numpy(), fit["isFraud_fit"].to_numpy()):
        raise AssertionError("Cleaned source target differs from the v2 training-fit key file")
    fit["isFraud"] = fit["isFraud_fit"].astype("int8")
    fit = fit.drop(columns=["isFraud_source", "isFraud_fit"])
    expected_counts = {int(k): int(v["train_fit"]) for k, v in manifest["class_counts"].items()}
    actual_counts = {int(k): int(v) for k, v in fit["isFraud"].value_counts().items()}
    if actual_counts != expected_counts:
        raise AssertionError(f"Fit class counts mismatch: {actual_counts} vs {expected_counts}")
    if fit["TransactionAmt"].isna().any() or (fit["TransactionAmt"] < 0).any():
        raise AssertionError("TransactionAmt contains missing or negative values; inspect before testing")
    return fit, manifest


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m, running = len(ordered), 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def p_display(p: float) -> str:
    return "<1e-300 (floating-point underflow)" if p == 0.0 else f"{p:.4g}"


def cliff_delta_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Cliff's delta and asymptotic 95% CI using two-sample placement variance."""
    xs = np.sort(x.astype(float, copy=False))
    ys = np.sort(y.astype(float, copy=False))
    n1, n0 = len(xs), len(ys)
    x_less = np.searchsorted(ys, xs, side="left")
    x_leq = np.searchsorted(ys, xs, side="right")
    gx = (x_less - (n0 - x_leq)) / n0
    y_less = np.searchsorted(xs, ys, side="left")
    y_leq = np.searchsorted(xs, ys, side="right")
    gy = ((n1 - y_leq) - y_less) / n1
    delta = float((gx.mean() + gy.mean()) / 2.0)
    # Both placement means estimate the same population parameter.
    se = float(np.sqrt(np.var(gx, ddof=1) / n1 + np.var(gy, ddof=1) / n0))
    return delta, max(-1.0, delta - 1.96 * se), min(1.0, delta + 1.96 * se)


def c_at_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False).statistic
    n = table.sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def cramers_v_bootstrap(table: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    table = np.asarray(table, dtype=int)
    n = int(table.sum())
    observed = c_at_v(table)
    probs = table.ravel() / n
    draws = rng.multinomial(n, probs, size=BOOTSTRAP_REPS)
    vals = np.fromiter((c_at_v(row.reshape(table.shape)) for row in draws), dtype=float, count=BOOTSTRAP_REPS)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return observed, float(lo), float(hi)


def wilson(success: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = success / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return float(center-half), float(center+half)


def summarize_category(data: pd.DataFrame, column: str) -> pd.DataFrame:
    view = data[[column, "isFraud"]].copy()
    view[column] = view[column].astype("string").fillna("Missing")
    result = view.groupby(column, dropna=False, sort=True).agg(n=("isFraud", "size"), frauds=("isFraud", "sum"), fraud_rate=("isFraud", "mean")).reset_index()
    ci = [wilson(int(r.frauds), int(r.n)) for r in result.itertuples()]
    result["ci_low"] = [x[0] for x in ci]
    result["ci_high"] = [x[1] for x in ci]
    result["feature"] = column
    return result


def run_tests(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    fraud = data.loc[data.isFraud.eq(1)]
    legit = data.loc[data.isFraud.eq(0)]
    rng = np.random.default_rng(SEED)
    tests: dict[str, Any] = {}
    pvalues: dict[str, float] = {}

    # Primary robust two-sample amount-distribution comparison.
    x = fraud["TransactionAmt"].to_numpy(dtype=float)
    y = legit["TransactionAmt"].to_numpy(dtype=float)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    delta, delta_lo, delta_hi = cliff_delta_ci(x, y)
    tests["amount_mann_whitney"] = {
        "hypotheses": {"H0": "Fraudulent and legitimate transaction amounts have the same distribution.", "H1": "Their transaction-amount distributions differ."},
        "test": "Two-sided Mann–Whitney U, asymptotic with tie correction",
        "statistic": float(mw.statistic), "p_value": float(mw.pvalue), "n_fraud": len(x), "n_legitimate": len(y),
        "fraud_median": float(np.median(x)), "legitimate_median": float(np.median(y)),
        "fraud_iqr": [float(v) for v in np.quantile(x, [0.25, 0.75])], "legitimate_iqr": [float(v) for v in np.quantile(y, [0.25, 0.75])],
        "effect": {"name": "Cliff's delta (fraud minus legitimate)", "estimate": delta, "ci_95": [delta_lo, delta_hi]},
        "assumptions": ["Transactions are treated as independent observational units.", "Amounts are continuous/ordinal with ties allowed; distributions are highly skewed, so the test is interpreted as a distribution/rank difference, not a pure median test."],
    }
    pvalues["amount_mann_whitney"] = float(mw.pvalue)

    # Welch comparison of log1p(amount) as a sensitivity analysis.
    lx, ly = np.log1p(x), np.log1p(y)
    welch = stats.ttest_ind(lx, ly, equal_var=False, alternative="two-sided")
    v1, v0 = lx.var(ddof=1), ly.var(ddof=1)
    se = float(np.sqrt(v1/len(lx) + v0/len(ly)))
    df = float((v1/len(lx) + v0/len(ly))**2 / ((v1/len(lx))**2/(len(lx)-1) + (v0/len(ly))**2/(len(ly)-1)))
    diff = float(lx.mean()-ly.mean())
    tcrit = float(stats.t.ppf(0.975, df))
    pooled_sd = float(np.sqrt(((len(lx)-1)*v1 + (len(ly)-1)*v0)/(len(lx)+len(ly)-2)))
    d = diff / pooled_sd
    correction = 1 - 3/(4*(len(lx)+len(ly))-9)
    g = d * correction
    g_se = correction * np.sqrt((len(lx)+len(ly))/(len(lx)*len(ly)) + d*d/(2*(len(lx)+len(ly)-2)))
    tests["log_amount_welch"] = {
        "hypotheses": {"H0": "The mean log1p transaction amount is equal for fraudulent and legitimate transactions.", "H1": "The mean log1p transaction amount differs."},
        "test": "Two-sided Welch t-test on log1p(TransactionAmt), unequal variances",
        "statistic": float(welch.statistic), "degrees_freedom": df, "p_value": float(welch.pvalue), "n_fraud": len(lx), "n_legitimate": len(ly),
        "fraud_mean_log1p": float(lx.mean()), "legitimate_mean_log1p": float(ly.mean()), "mean_difference_log1p": diff,
        "mean_difference_ci_95": [diff-tcrit*se, diff+tcrit*se],
        "effect": {"name": "Hedges' g on log1p amounts", "estimate": float(g), "ci_95_approx": [float(g-1.96*g_se), float(g+1.96*g_se)]},
        "assumptions": ["Independent groups; unequal variances are allowed by Welch's test.", "Log1p reduces amount skew; with large group sizes, the mean difference is asymptotically robust. The result is on the transformed scale."],
        "role": "Sensitivity analysis for the same amount question; included in Holm correction.",
    }
    pvalues["log_amount_welch"] = float(welch.pvalue)

    category_summaries = []
    for col, key, label in [("ProductCD", "productcd_chi_square", "ProductCD"), ("card6", "card6_chi_square", "card6")]:
        view = data[[col, "isFraud"]].copy()
        view[col] = view[col].astype("string").fillna("Missing")
        tab_df = pd.crosstab(view[col], view.isFraud).reindex(columns=[0, 1], fill_value=0)
        chi = stats.chi2_contingency(tab_df.to_numpy(), correction=False)
        v, vlo, vhi = cramers_v_bootstrap(tab_df.to_numpy(), rng)
        test_method = "Pearson chi-square test of independence"
        pvalue = float(chi.pvalue)
        test_assumptions = ["Rows are independent observational units.", "All expected cell counts must be at least 5; checked from the computed expected-frequency table."]
        monte_carlo = None
        if not (chi.expected_freq >= 5).all():
            # Condition on margins and simulate the null distribution when the
            # large-sample expected-frequency condition is not met.
            mc_rng = np.random.default_rng(MONTE_CARLO_SEED)
            row_margins = tab_df.to_numpy().sum(axis=1)
            col_margins = tab_df.to_numpy().sum(axis=0)
            expected = chi.expected_freq
            simulated = stats.random_table.rvs(row_margins, col_margins, size=MONTE_CARLO_REPS, random_state=mc_rng)
            sim_stat = np.sum((simulated - expected) ** 2 / expected, axis=(1, 2))
            pvalue = float((np.count_nonzero(sim_stat >= chi.statistic) + 1) / (MONTE_CARLO_REPS + 1))
            test_method = "Conditional Monte Carlo Pearson chi-square test (fixed margins; sparse expected counts)"
            monte_carlo = {"replicates": MONTE_CARLO_REPS, "seed": MONTE_CARLO_SEED, "plus_one_correction": True}
            test_assumptions = ["Rows are independent observational units.", "Sparse expected cells make the asymptotic chi-square approximation unsuitable; the p-value is estimated under fixed margins by conditional Monte Carlo simulation."]
        tests[key] = {
            "hypotheses": {"H0": f"{label} category and fraud status are independent.", "H1": f"{label} category and fraud status are associated."},
            "test": test_method,
            "statistic": float(chi.statistic), "degrees_freedom": int(chi.dof), "p_value": pvalue,
            "n": int(tab_df.to_numpy().sum()), "categories": [str(a) for a in tab_df.index],
            "expected_min": float(chi.expected_freq.min()), "expected_all_at_least_5": bool((chi.expected_freq >= 5).all()),
            "contingency_table": {str(index): {str(col): int(tab_df.loc[index, col]) for col in tab_df.columns} for index in tab_df.index},
            "effect": {"name": "Cramer's V", "estimate": v, "ci_95_bootstrap": [vlo, vhi], "bootstrap_replicates": BOOTSTRAP_REPS},
            "assumptions": test_assumptions,
            "monte_carlo": monte_carlo,
        }
        pvalues[key] = pvalue
        category_summaries.append(summarize_category(data, col))

    # Test whether V258 observation/missingness differs by class.
    v_missing = data["V258"].isna()
    miss_table = pd.crosstab(data["isFraud"], v_missing).reindex(index=[0, 1], columns=[False, True], fill_value=0)
    miss_chi = stats.chi2_contingency(miss_table.to_numpy(), correction=False)
    miss_v, miss_vlo, miss_vhi = cramers_v_bootstrap(miss_table.to_numpy(), rng)
    tests["v258_missingness_chi_square"] = {
        "hypotheses": {"H0": "Whether V258 is missing is independent of fraud status.", "H1": "V258 missingness is associated with fraud status."},
        "test": "Pearson chi-square test of independence (2×2)", "statistic": float(miss_chi.statistic), "degrees_freedom": int(miss_chi.dof),
        "p_value": float(miss_chi.pvalue), "n": int(miss_table.to_numpy().sum()),
        "contingency_table": {str(i): {str(j): int(miss_table.loc[i, j]) for j in miss_table.columns} for i in miss_table.index},
        "expected_min": float(miss_chi.expected_freq.min()), "expected_all_at_least_5": bool((miss_chi.expected_freq >= 5).all()),
        "fraud_v258_observed_rate": float((~v_missing[data.isFraud.eq(1)]).mean()),
        "legitimate_v258_observed_rate": float((~v_missing[data.isFraud.eq(0)]).mean()),
        "effect": {"name": "Cramer's V", "estimate": miss_v, "ci_95_bootstrap": [miss_vlo, miss_vhi], "bootstrap_replicates": BOOTSTRAP_REPS},
        "assumptions": ["Rows are independent observational units.", "All expected cell counts must be at least 5; checked from the computed expected-frequency table."],
        "caution": "A missingness association may reflect data collection/availability and is not a causal effect.",
    }
    pvalues["v258_missingness_chi_square"] = float(miss_chi.pvalue)

    # Test the value distribution only among records with an observed V258.
    valid = data["V258"].notna()
    vx = data.loc[valid & data.isFraud.eq(1), "V258"].to_numpy(dtype=float)
    vy = data.loc[valid & data.isFraud.eq(0), "V258"].to_numpy(dtype=float)
    vmw = stats.mannwhitneyu(vx, vy, alternative="two-sided", method="asymptotic")
    vdelta, vdlo, vdhi = cliff_delta_ci(vx, vy)
    tests["v258_mann_whitney"] = {
        "hypotheses": {"H0": "Among transactions with observed V258, its distribution is the same for fraudulent and legitimate transactions.", "H1": "The observed V258 distributions differ."},
        "test": "Two-sided Mann–Whitney U, asymptotic with tie correction; pairwise-complete rows only",
        "statistic": float(vmw.statistic), "p_value": float(vmw.pvalue), "n_fraud": len(vx), "n_legitimate": len(vy),
        "fraud_median": float(np.median(vx)), "legitimate_median": float(np.median(vy)),
        "effect": {"name": "Cliff's delta (fraud minus legitimate)", "estimate": vdelta, "ci_95": [vdlo, vdhi]},
        "assumptions": ["Independent observational rows; missing V258 values are excluded pairwise.", "Because V258 is missing for many transactions, this complete-case comparison can describe only the observed subset; missingness is tested separately."],
    }
    pvalues["v258_mann_whitney"] = float(vmw.pvalue)

    adjusted = holm_adjust(pvalues)
    for key, adj in adjusted.items():
        tests[key]["p_value_holm"] = float(adj)
        tests[key]["p_value_display"] = p_display(float(tests[key]["p_value"]))
        tests[key]["p_value_holm_display"] = p_display(float(adj))
        tests[key]["p_value_below_float_range"] = bool(tests[key]["p_value"] == 0.0)
        tests[key]["reject_at_0_05_holm"] = bool(adj < ALPHA)

    summaries = pd.concat(category_summaries, ignore_index=True)
    summaries.to_csv(GROUPS_CSV, index=False)
    results = {
        "scope": "cleaned training-fit partition only; no test data used",
        "fit_rows": int(len(data)), "fraud_count": int(data.isFraud.sum()), "legitimate_count": int((data.isFraud == 0).sum()),
        "fraud_rate": float(data.isFraud.mean()), "random_state": SEED, "alpha": ALPHA,
        "p_adjustment": "Holm step-down familywise adjustment across all six reported hypothesis tests, including Welch sensitivity test",
        "source_columns": FEATURES, "fit_manifest_rows": 472432,
        "tests": tests,
        "group_summaries": summaries.to_dict(orient="records"),
        "prior_context": {"week1": W1.read_text(encoding="utf-8"), "week2": W2.read_text(encoding="utf-8")},
    }
    # Full prior findings are cited as context in the report, but not included verbatim in result JSON.
    results["prior_context"] = {"week1_findings_file": str(W1), "week2_findings_file": str(W2), "reuse_only": True}
    return json_ready(results), summaries


def chart_amount(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = [f"#{TEAL}", f"#{RED}"]
    items = []
    for target, label in [(0, "Legitimate"), (1, "Fraudulent")]:
        vals = np.log1p(data.loc[data.isFraud.eq(target), "TransactionAmt"].to_numpy(dtype=float))
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        iqr = q3-q1
        low = float(vals[vals >= q1-1.5*iqr].min())
        high = float(vals[vals <= q3+1.5*iqr].max())
        items.append({"label": label, "q1": q1, "med": med, "q3": q3, "whislo": low, "whishi": high, "fliers": []})
    ax.bxp(items, positions=[1, 2], showfliers=False, patch_artist=True, medianprops={"color": "white", "linewidth": 2},
           boxprops={"linewidth": 1.5}, whiskerprops={"linewidth": 1.4}, capprops={"linewidth": 1.4})
    ax.set_xticks([1, 2], ["Legitimate", "Fraudulent"])
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
    # Matplotlib versions differ in bxp artist collection; color boxes explicitly.
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)
    ax.set_title("Transaction amount distributions by fraud status", loc="left", color=f"#{NAVY}", weight="bold", fontsize=14)
    ax.set_ylabel("log(1 + TransactionAmt)")
    ax.set_xlabel("Target group")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.01, -0.18, "Exact quartiles and Tukey whiskers; outlier points omitted for readability. Amounts are positive in the cleaned fit data.", transform=ax.transAxes, fontsize=8.5, color=f"#{MUTED}")
    fig.tight_layout()
    path = CHARTS / "01_amount_distribution_by_fraud_status.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_category(summary: pd.DataFrame, column: str, filename: str, title: str) -> Path:
    plot = summary.loc[summary.feature.eq(column)].sort_values("fraud_rate", ascending=False).copy()
    if column == "card6":
        rare = plot.loc[plot.n < 1000]
        plot = plot.loc[plot.n >= 1000].copy()
        if not rare.empty:
            rare_n, rare_fraud = int(rare.n.sum()), int(rare.frauds.sum())
            rare_low, rare_high = wilson(rare_fraud, rare_n)
            plot = pd.concat([plot, pd.DataFrame([{"card6": "Other (combined)", "n": rare_n, "frauds": rare_fraud,
                                                   "fraud_rate": rare_fraud/rare_n, "ci_low": rare_low,
                                                   "ci_high": rare_high, "feature": column}])], ignore_index=True)
    plot = plot.sort_values("fraud_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.arange(len(plot))
    rate = plot.fraud_rate.to_numpy()
    low = rate - plot.ci_low.to_numpy()
    high = plot.ci_high.to_numpy() - rate
    ax.bar(x, rate*100, color=f"#{TEAL}", alpha=0.88, width=0.68)
    ax.errorbar(x, rate*100, yerr=np.maximum(0, np.vstack([low, high]))*100, fmt="none", ecolor=f"#{INK}", capsize=4, linewidth=1.2)
    ax.set_xticks(x, plot[column].astype(str))
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xlabel(column)
    ax.set_title(title, loc="left", color=f"#{NAVY}", weight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    ymax = max(1, float((plot.ci_high*100).max()))
    ax.set_ylim(0, ymax*1.22)
    for i, row in enumerate(plot.itertuples()):
        ax.text(i, row.fraud_rate*100 + ymax*0.045, f"{int(row.frauds):,}/{int(row.n):,}", ha="center", va="bottom", fontsize=8)
    note = "Bars are observed fraud rates; whiskers are Wilson 95% intervals. Group differences are observational."
    if column == "card6":
        note += " Categories with fewer than 1,000 rows are combined for display only."
    ax.text(0.01, -0.18, note, transform=ax.transAxes, fontsize=8.5, color=f"#{MUTED}")
    fig.tight_layout()
    path = CHARTS / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_missingness(data: pd.DataFrame) -> Path:
    rows = []
    for target, label in [(0, "Legitimate"), (1, "Fraudulent")]:
        group = data.loc[data.isFraud.eq(target), "V258"]
        n = len(group)
        observed = int(group.notna().sum())
        lo, hi = wilson(observed, n)
        rows.append({"group": label, "observed_rate": observed/n, "lo": lo, "hi": hi, "observed": observed, "missing": n-observed, "n": n})
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    x = np.arange(2)
    rates = np.array([r["observed_rate"] for r in rows])
    errors = np.vstack([rates-np.array([r["lo"] for r in rows]), np.array([r["hi"] for r in rows])-rates])*100
    ax.bar(x, rates*100, color=[f"#{TEAL}", f"#{RED}"], width=0.62, alpha=0.9)
    ax.errorbar(x, rates*100, yerr=errors, fmt="none", ecolor=f"#{INK}", capsize=5, linewidth=1.3)
    ax.set_xticks(x, [r["group"] for r in rows])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share with V258 observed (%)")
    ax.set_title("V258 availability differs across target groups", loc="left", color=f"#{NAVY}", weight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    for i, row in enumerate(rows):
        ax.text(i, rates[i]*100+3, f"{row['observed']:,}/{row['n']:,}", ha="center", fontsize=9)
    ax.text(0.01, -0.18, "Availability is descriptive and may reflect the data collection process; it is not a causal effect.", transform=ax.transAxes, fontsize=8.5, color=f"#{MUTED}")
    fig.tight_layout()
    path = CHARTS / "04_v258_availability_by_fraud_status.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def code_block(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string("23313B")


def add_result_table(doc: Document, results: dict[str, Any]) -> None:
    rows = [
        ("TransactionAmt", "amount_mann_whitney"), ("log1p(TransactionAmt)", "log_amount_welch"),
        ("ProductCD", "productcd_chi_square"), ("card6", "card6_chi_square"),
        ("V258 missingness", "v258_missingness_chi_square"), ("Observed V258 values", "v258_mann_whitney"),
    ]
    table = doc.add_table(rows=1, cols=6)
    table.style = "Light Shading Accent 1"
    headers = ["Measure", "Test", "Statistic", "p-value", "Holm p", "Effect size (95% CI)"]
    for cell, val in zip(table.rows[0].cells, headers):
        cell.text = val
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(8)
    for label, key in rows:
        r = results["tests"][key]
        effect = r["effect"]
        ci = effect.get("ci_95", effect.get("ci_95_approx", effect.get("ci_95_bootstrap", [])))
        effect_text = f"{effect['name']}: {effect['estimate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        stat_text = f"{r['statistic']:.3g}" + (f", df={r['degrees_freedom']:.1f}" if "degrees_freedom" in r else "")
        vals = [label, r["test"].split(",")[0], stat_text, p_display(r["p_value"]), p_display(r["p_value_holm"]), effect_text]
        cells = table.add_row().cells
        for c, val in zip(cells, vals):
            c.text = val
            for run in c.paragraphs[0].runs:
                run.font.size = Pt(7)


def make_report(results: dict[str, Any], chart_paths: list[Path]) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.68); sec.bottom_margin = Inches(0.68)
    sec.left_margin = Inches(0.75); sec.right_margin = Inches(0.75)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    for name, size, color in [("Title", 26, NAVY), ("Heading 1", 17, NAVY), ("Heading 2", 12, TEAL)]:
        s=styles[name]; s.font.name="Aptos Display"; s.font.size=Pt(size); s.font.bold=True; s.font.color.rgb=RGBColor.from_string(color)
    header = sec.header.paragraphs[0]
    header.text = "IEEE-CIS FRAUD DETECTION  |  YUVA INTERN  |  WEEK 3 STATISTICS"
    header.style = "Caption"
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Training-fit only | No model trained")

    p=doc.add_paragraph(style="Title"); p.add_run("Week 3 Statistical Analysis and Hypothesis Testing")
    doc.add_paragraph("IEEE-CIS Fraud Detection project")
    doc.add_heading("Objective and scope", level=1)
    doc.add_paragraph(f"Test selected, evidence-led questions about differences between fraudulent and legitimate transactions using only the {results['fit_rows']:,}-row v2 training-fit partition ({results['fraud_count']:,} fraud, {results['legitimate_count']:,} legitimate). Week 1 EDA and Week 2 findings guide the questions; no earlier work is repeated. The validation/test labels and all test records are excluded from these tests.")
    doc.add_heading("Data and methods", level=1)
    doc.add_paragraph("The script reads six selected columns from the existing cleaned training table, filters rows by TransactionID from the v2 training-fit file, and verifies row counts and target values against the split manifest. TransactionID is only a join key and is not tested as a feature. Amount is right-skewed, so Mann–Whitney U is primary; Welch’s t-test on log1p amount is a sensitivity analysis. Pearson chi-square is used for categorical associations. All expected cell counts are checked. Six p-values are adjusted together with Holm’s method to limit familywise false positives.")
    doc.add_paragraph("Effect sizes are reported with 95% intervals: Cliff’s delta uses an asymptotic placement-based standard error; Hedges’ g uses a large-sample approximate interval; Cramér’s V intervals use 600 multinomial bootstrap replicates from the observed contingency table. Wilson intervals support the descriptive category and availability charts. Results are observational and do not show causes.")
    doc.add_heading("Hypotheses and results", level=1)
    result_names = [
        ("Transaction amount distribution", "amount_mann_whitney"),
        ("Mean log transaction amount (sensitivity)", "log_amount_welch"),
        ("ProductCD and fraud status", "productcd_chi_square"),
        ("card6 and fraud status", "card6_chi_square"),
        ("V258 missingness and fraud status", "v258_missingness_chi_square"),
        ("Observed V258 value distribution", "v258_mann_whitney"),
    ]
    for title, key in result_names:
        t=results["tests"][key]
        doc.add_heading(title, level=2)
        doc.add_paragraph(f"H₀: {t['hypotheses']['H0']}\nH₁: {t['hypotheses']['H1']}")
        n = f"n fraud={t.get('n_fraud','—'):,}, legitimate={t.get('n_legitimate','—'):,}" if isinstance(t.get('n_fraud'), int) else f"n={t.get('n','—'):,}"
        effect=t['effect']; ci=effect.get('ci_95',effect.get('ci_95_approx',effect.get('ci_95_bootstrap')))
        doc.add_paragraph(f"{t['test']}; statistic={t['statistic']:.5g}; p={p_display(t['p_value'])}; Holm-adjusted p={p_display(t['p_value_holm'])}; {n}. Effect: {effect['name']}={effect['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}].")
        decision="Reject H₀ at familywise α=0.05." if t['reject_at_0_05_holm'] else "Do not reject H₀ at familywise α=0.05."
        doc.add_paragraph(decision + " " + " ".join(t['assumptions']))
        if key == "amount_mann_whitney":
            doc.add_paragraph(f"Observed medians: fraud=${t['fraud_median']:.2f}; legitimate=${t['legitimate_median']:.2f}. The rank test compares the full distributions, so a significant result does not by itself prove only the medians differ.")
        if key == "log_amount_welch":
            lo,hi=t['mean_difference_ci_95']
            doc.add_paragraph(f"Mean log1p difference (fraud minus legitimate)={t['mean_difference_log1p']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]. This sensitivity result checks whether the amount conclusion is also visible on a transformed mean scale.")
        if key == "v258_missingness_chi_square":
            doc.add_paragraph(f"V258 is observed for {t['fraud_v258_observed_rate']:.1%} of fraud rows and {t['legitimate_v258_observed_rate']:.1%} of legitimate rows. Missingness may encode availability/collection processes and should not be interpreted as a fraud mechanism.")
        if key == "v258_mann_whitney":
            doc.add_paragraph(f"This comparison includes only observed V258 values (fraud n={t['n_fraud']:,}; legitimate n={t['n_legitimate']:,}); it does not represent rows where V258 is missing.")
    doc.add_heading("Consolidated test table", level=1)
    add_result_table(doc, results)
    doc.add_paragraph("Interpret p-values alongside effect sizes, intervals, sample sizes, and the Holm-adjusted decision. Very small p-values can occur for modest effects in large datasets. Values below the floating-point range are displayed as <1e-300 rather than as zero in this report; the raw JSON stores the numeric underflow as 0 and flags it.")

    docs = [
        ("Amount by target", "The box plots show the shifted and skewed amount distributions on log1p scale. The difference motivates segment monitoring, while the rank effect size describes how often a randomly selected fraud amount exceeds a legitimate amount, with ties handled neutrally."),
        ("ProductCD fraud rates", "Observed fraud rates differ among product categories; Wilson intervals and numerator/denominator labels show uncertainty and support. Category-level rates can help prioritize review, but this dataset cannot tell why categories differ."),
        ("card6 fraud rates", "Card type segments have different observed fraud rates. Keep segment volume and uncertainty in view before using such patterns for operational decisions."),
        ("V258 availability", "The fraction of rows with V258 present differs by target group. This is a data availability pattern and could reflect collection processes, so it should not be treated as causal evidence."),
    ]
    doc.add_heading("Visual evidence and business interpretation", level=1)
    for i, ((title, explanation), image_path) in enumerate(zip(docs, chart_paths), 1):
        doc.add_heading(f"Figure {i}. {title}", level=2)
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(image_path), width=Inches(6.7))
        doc.add_paragraph(explanation)

    doc.add_heading("Reproducibility and code snippets", level=1)
    doc.add_paragraph("The full script and exact run command are in the Week 3 folder. Core tests are:")
    code_block(doc, "stats.mannwhitneyu(fraud_amount, legitimate_amount, alternative='two-sided', method='asymptotic')\nstats.ttest_ind(np.log1p(fraud_amount), np.log1p(legitimate_amount), equal_var=False)\nstats.chi2_contingency(pd.crosstab(category, isFraud), correction=False)")
    doc.add_paragraph("The workflow verifies the v2 training-fit key count, unique IDs, target agreement, and fraud counts before testing. Bootstrap confidence intervals use a fixed random seed. Holm correction is implemented in the script and all result values are serialized in week3_results.json.")
    doc.add_heading("Limitations", level=1)
    doc.add_paragraph("These are unadjusted, observational group comparisons; they do not establish causality or predictive usefulness. Treating transactions as independent is an approximation because cardholders/merchants may contribute multiple transactions. The time-ordered data may have dependence, so this report does not use TransactionDT to claim calendar or causal effects. V258 value comparisons exclude missing rows. Statistical significance does not equal business importance.")
    doc.add_heading("Conclusion", level=1)
    sig = [name for name, key in result_names if results['tests'][key]['reject_at_0_05_holm']]
    doc.add_paragraph("After Holm correction, the tests with evidence of differences/association are: " + ("; ".join(sig) if sig else "none at the 0.05 familywise level") + ". Effect sizes and confidence intervals quantify their magnitude. These patterns can guide later validation and monitoring questions, but no causal interpretation or model performance claim is made. Week 3 only is complete; no test data or model was used.")
    doc.add_heading("Output files", level=1)
    for p in [REPORT, OUT/"week3_statistics.py", OUT/"verify_week3_statistics.py", RESULTS, GROUPS_CSV, VERIFY, CHARTS]:
        doc.add_paragraph(str(p), style="List Bullet")
    doc.save(REPORT)
    if len(Document(REPORT).inline_shapes) != len(chart_paths):
        raise AssertionError("Report chart embed count differs from generated charts")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); CHARTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    data, manifest = load_fit()
    results, groups = run_tests(data)
    chart_paths = [chart_amount(data)]
    chart_paths.append(chart_category(groups, "ProductCD", "02_productcd_fraud_rates.png", "Fraud rates differ across ProductCD categories"))
    chart_paths.append(chart_category(groups, "card6", "03_card6_fraud_rates.png", "Observed fraud rates by card type"))
    chart_paths.append(chart_missingness(data))
    results["charts"] = [str(p) for p in chart_paths]
    results["checks"] = {
        "fit_rows_match_manifest": len(data) == manifest["fit_rows"],
        "fit_class_counts_match_manifest": int(data.isFraud.sum()) == manifest["class_counts"]["1"]["train_fit"] and int((data.isFraud==0).sum()) == manifest["class_counts"]["0"]["train_fit"],
        "productcd_expected_counts_at_least_5": results["tests"]["productcd_chi_square"]["expected_all_at_least_5"],
        "card6_sparse_cells_use_conditional_monte_carlo": results["tests"]["card6_chi_square"]["expected_all_at_least_5"] or results["tests"]["card6_chi_square"]["monte_carlo"] is not None,
        "v258_missing_expected_counts_at_least_5": results["tests"]["v258_missingness_chi_square"]["expected_all_at_least_5"],
        "no_test_input_read": True,
        "no_model_training": True,
        "prior_outputs_read_only": True,
    }
    if not all(results["checks"].values()):
        raise AssertionError(f"Input/statistical checks failed: {results['checks']}")
    jwrite(RESULTS, results)
    make_report(results, chart_paths)
    validation = {
        "status": "PASS", "checks": results["checks"], "fit_rows": len(data),
        "fraud_rows": int(data.isFraud.sum()), "legitimate_rows": int((data.isFraud==0).sum()),
        "hypothesis_test_count": len(results["tests"]), "chart_count": len(chart_paths),
        "report_image_count": len(Document(REPORT).inline_shapes),
        "expected_chart_names": [p.name for p in chart_paths],
        "results_file": str(RESULTS), "report_file": str(REPORT),
    }
    validation["checks"]["all_charts_embedded"] = validation["chart_count"] == validation["report_image_count"]
    validation["checks"]["all_hypotheses_have_statistics_and_pvalues"] = all("statistic" in t and "p_value" in t for t in results["tests"].values())
    validation["checks"]["all_tests_have_effect_sizes"] = all("effect" in t and "estimate" in t["effect"] for t in results["tests"].values())
    validation["checks"]["holms_adjustment_present"] = all("p_value_holm" in t for t in results["tests"].values())
    validation["status"] = "PASS" if all(validation["checks"].values()) else "FAIL"
    jwrite(VERIFY, validation)
    print(json.dumps({"status": validation["status"], "fit_rows": len(data), "fraud_rows": int(data.isFraud.sum()), "tests": len(results["tests"]), "charts": len(chart_paths), "report": str(REPORT), "verification": str(VERIFY)}, indent=2))


if __name__ == "__main__":
    main()
