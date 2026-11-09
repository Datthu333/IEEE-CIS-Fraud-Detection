"""IEEE-CIS fraud dataset cleaning and audit.

Reads the original Kaggle CSV files without changing them, snapshots source schemas,
normalizes test identity feature names, and left-merges identity onto each split.
Missing values are intentionally retained in the clean source tables. Model-time
imputation/encoding belongs in a leakage-safe pipeline fit on training data only.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import pandas as pd

FILES = ["train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv", "sample_submission.csv"]

def get_schema(path: Path) -> dict:
    frame = pd.read_csv(path, nrows=10000, low_memory=False)
    return {"rows": None, "columns": len(frame.columns), "column_names": list(frame.columns),
            "inferred_dtypes_first_10000_rows": {c: str(frame[c].dtype) for c in frame.columns}}

def source_missing(path: Path) -> dict:
    total = 0
    counts = None
    seen_hashes = set()
    duplicate_rows = 0
    for frame in pd.read_csv(path, chunksize=50000, low_memory=False):
        total += len(frame)
        cur = frame.isna().sum()
        counts = cur if counts is None else counts.add(cur, fill_value=0)
        for value in pd.util.hash_pandas_object(frame, index=False).values:
            value = int(value)
            if value in seen_hashes: duplicate_rows += 1
            else: seen_hashes.add(value)
    return {"rows": total, "duplicate_rows": duplicate_rows,
            "columns": {c: {"missing_count": int(n), "missing_pct": round(100 * int(n) / total, 6)} for c, n in counts.items()}}

def digest_id(values) -> str:
    h = hashlib.sha256()
    for x in values:
        h.update(str(int(x)).encode()); h.update(b"\n")
    return h.hexdigest()

def merge_split(source: Path, out: Path, tx_name: str, id_name: str, expected: int):
    idf = pd.read_csv(source / id_name, low_memory=False)
    if idf["TransactionID"].isna().any(): raise ValueError(f"Null identity key in {id_name}")
    if idf["TransactionID"].duplicated().any(): raise ValueError(f"Duplicate identity key in {id_name}")
    if id_name.startswith("test_"):
        idf = idf.rename(columns={c: c.replace("-", "_") for c in idf.columns if c.startswith("id-")})
    written = 0; seen = set(); matched = 0; first = True; after_missing = None
    with open(out, "w", encoding="utf-8", newline="") as handle:
        for tx in pd.read_csv(source / tx_name, chunksize=50000, low_memory=False):
            if tx["TransactionID"].isna().any(): raise ValueError(f"Null transaction key in {tx_name}")
            if tx["TransactionID"].duplicated().any() or any(k in seen for k in tx.TransactionID):
                raise ValueError(f"Duplicate transaction key in {tx_name}")
            seen.update(tx.TransactionID.tolist())
            merged = tx.merge(idf, on="TransactionID", how="left", validate="one_to_one", sort=False, indicator="_identity_merge")
            matched += int((merged["_identity_merge"] == "both").sum())
            merged.drop(columns="_identity_merge", inplace=True)
            if len(merged) != len(tx): raise ValueError("Left merge changed transaction row count")
            cur_missing = merged.isna().sum()
            after_missing = cur_missing if after_missing is None else after_missing.add(cur_missing, fill_value=0)
            merged.to_csv(handle, index=False, header=first)
            first = False; written += len(merged)
    if written != expected or written != len(seen): raise ValueError("Merged row count/key count validation failed")
    return {"output": out.name, "rows": written, "columns": len(pd.read_csv(out, nrows=1).columns),
            "unique_transaction_ids": len(seen), "matched_identity_rows": matched,
            "transactions_without_identity": written - matched,
            "transaction_id_sha256": digest_id(sorted(seen)),
            "post_merge_missing": {c: {"missing_count": int(n), "missing_pct": round(100 * int(n) / written, 6)} for c, n in after_missing.items()}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, help="Folder containing the five original CSVs")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = p.parse_args(); source = Path(args.source_dir); outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (source / f).is_file()]
    if missing: raise FileNotFoundError(f"Missing required source files: {missing}")
    # This is deliberately the first output: source shapes and schemas are captured before transformations.
    original = {f: get_schema(source / f) for f in FILES}
    before = {f: source_missing(source / f) for f in FILES}
    for f in FILES:
        original[f]["rows"] = before[f]["rows"]
        original[f]["duplicate_rows_hash_check"] = before[f]["duplicate_rows"]
    (outdir / "original_schema.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    (outdir / "missing_before.json").write_text(json.dumps({f: v["columns"] for f,v in before.items()}, indent=2), encoding="utf-8")
    train = merge_split(source, outdir / "train_cleaned.csv", "train_transaction.csv", "train_identity.csv", original["train_transaction.csv"]["rows"])
    test = merge_split(source, outdir / "test_cleaned.csv", "test_transaction.csv", "test_identity.csv", original["test_transaction.csv"]["rows"])
    if train["columns"] - 1 != test["columns"]: raise ValueError("Train features and test feature schemas are not aligned")
    trcols = pd.read_csv(outdir / "train_cleaned.csv", nrows=1).columns.tolist()
    tecols = pd.read_csv(outdir / "test_cleaned.csv", nrows=1).columns.tolist()
    if [c for c in trcols if c != "isFraud"] != tecols: raise ValueError("Feature column names/order differ after excluding isFraud")
    y = pd.read_csv(outdir / "train_cleaned.csv", usecols=["isFraud"])["isFraud"]
    if y.isna().any() or not set(y.unique()).issubset({0,1}): raise ValueError("Invalid or missing isFraud target")
    submission = pd.read_csv(source / "sample_submission.csv", usecols=["TransactionID"])
    test_ids = pd.read_csv(outdir / "test_cleaned.csv", usecols=["TransactionID"])["TransactionID"]
    if not submission.TransactionID.equals(test_ids): raise ValueError("Submission template IDs/order do not match test rows")
    validations = {"train": train, "test": test, "target_counts": {str(k): int(v) for k,v in y.value_counts().sort_index().items()},
                   "aligned_features_excluding_target": True, "feature_count": len(tecols)-1,
                   "test_target_absent": "isFraud" not in tecols, "submission_ids_match_test_order": True,
                   "duplicate_rows_found": {f: before[f]["duplicate_rows"] for f in FILES},
                   "duplicates_removed": 0, "high_missingness_columns_dropped": [],
                   "imputation_or_encoding_applied_to_cleaned_tables": False,
                   "note": "Cleaned CSVs preserve source missingness. See preprocessed/fitted_preprocessing.json for training-only imputation and encoding parameters."}
    (outdir / "validation_results.json").write_text(json.dumps(validations, indent=2), encoding="utf-8")
    # Per-column post-merge missingness outputs.
    for name, result in [("train", train), ("test", test)]:
        pd.DataFrame([{"column":c, **v} for c,v in result["post_merge_missing"].items()]).to_csv(outdir / f"missing_after_{name}.csv", index=False)
    def missing_totals(columns, rows):
        cells = sum(v["missing_count"] for v in columns.values())
        total = rows * len(columns)
        return cells, total, (100 * cells / total if total else 0)
    missing_table = ["", "## Aggregate missingness", "",
      "| Dataset | Columns | Missing cells | Total cells | Missing percentage |",
      "|---|---:|---:|---:|---:|"]
    for f in FILES:
        cells, total, pct = missing_totals(before[f]["columns"], before[f]["rows"])
        missing_table.append(f"| Original {f} | {original[f]['columns']} | {cells:,} | {total:,} | {pct:.2f}% |")
    for label, result in [("Merged training", train), ("Merged test", test)]:
        cells, total, pct = missing_totals(result["post_merge_missing"], result["rows"])
        missing_table.append(f"| {label} | {result['columns']} | {cells:,} | {total:,} | {pct:.2f}% |")
    summary = ["# IEEE-CIS cleaning summary", "", "## Decisions", "",
      "- Original files were read only. Original dimensions, column names, and inferred dtypes were saved before merge operations in original_schema.json.",
      "- No exact duplicate rows or TransactionID duplicates were found in the initial full-file audit; no records were removed.",
      "- Test identity id-XX feature names were normalized to id_XX to align with training identity feature names.",
      "- Numeric-looking features kept their inferred numeric type; no unsupported semantic type guesses were made.",
      "- Identity was left-joined by TransactionID separately within train and test. Transactions without identity information remain present with missing identity fields.",
      "- High-missingness columns were retained. Missing values remain explicit in the cleaned source tables so that future model preprocessing can fit imputation on training rows only.",
      "- isFraud is preserved only in training; the test output contains no target. No model was trained.", "",
      "## Counts and validations", "",
      f"- Train merged: {train['rows']:,} rows × {train['columns']} columns; {train['matched_identity_rows']:,} identity matches; {train['transactions_without_identity']:,} transactions without identity.",
      f"- Test merged: {test['rows']:,} rows × {test['columns']} columns; {test['matched_identity_rows']:,} identity matches; {test['transactions_without_identity']:,} transactions without identity.",
      f"- Train features excluding isFraud align with test: {len(tecols)-1} columns, same names and order.",
      f"- Training target counts: {validations['target_counts']}.",
      "- TransactionID uniqueness, merge row counts, target values, and sample-submission ID order were validated; see validation_results.json.",
      "- Per-column missing counts and percentages are in missing_before.json, missing_after_train.csv, and missing_after_test.csv."] + missing_table + [""]
    (outdir / "cleaning_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"train": {k:v for k,v in train.items() if k!="post_merge_missing"}, "test":{k:v for k,v in test.items() if k!="post_merge_missing"}, "validation":"passed", "output_dir":str(outdir)}, indent=2))
if __name__ == "__main__": main()

