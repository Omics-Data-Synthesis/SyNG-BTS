"""Consolidate 24 TCGA miRNA datasets into one HDF5 file per project."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FULL_DATA_DIR = Path(__file__).resolve().parent.parent / "full_data"
RAW_DIR = FULL_DATA_DIR / "raw_real_data"
PROCESSED_DIR = FULL_DATA_DIR / "processed"
SYNTHETIC_DIR = FULL_DATA_DIR / "synthetic_output"
OUTPUT_DIR = FULL_DATA_DIR / "hdf5_output"

NORMS = ["raw_norm", "TC", "DESeq"]
MODELS = ["CVAE1_5", "CVAE1_10", "CVAE1_20"]
N_RAW_FEATURES = 1881

NORM_SUFFIXES: dict[str, str] = {
    "raw_norm": "_filtered.csv",
    "TC": "_filtered_TC.csv",
    "DESeq": "_filtered_DESeq.csv",
}

SYNTH_NORM_MAP: dict[str, str] = {"raw": "raw_norm", "TC": "TC", "DESeq": "DESeq"}
SYNTH_NORM_REVERSE: dict[str, str] = {v: k for k, v in SYNTH_NORM_MAP.items()}

MODEL_SUMMARY_TO_KEY: dict[str, str] = {
    "CVAE1-5": "CVAE1_5",
    "CVAE1-10": "CVAE1_10",
    "CVAE1-20": "CVAE1_20",
}

NORM_ATTRS: dict[str, dict[str, str]] = {
    "raw_norm": {"normalization_method": "log2_filtered", "transform": "log2(x+1)"},
    "TC": {"normalization_method": "TC_CPM", "transform": "log2(x+1)"},
    "DESeq": {
        "normalization_method": "DESeq_median_of_ratios",
        "transform": "log2(x+1)",
    },
}

SYNTH_SHARED_ATTRS: dict[str, int | float | bool] = {
    "epoch": 3000,
    "early_stop_patience": 20,
    "batch_frac": 0.1,
    "learning_rate": 0.0005,
    "random_seed": 42,
    "new_size": 1000,
    "apply_log": False,
}

MODEL_PARAMS: dict[str, dict[str, int]] = {
    "CVAE1_5": {"kl_weight": 5, "reconstruction_term_weight": 1},
    "CVAE1_10": {"kl_weight": 10, "reconstruction_term_weight": 1},
    "CVAE1_20": {"kl_weight": 20, "reconstruction_term_weight": 1},
}

HDF5_VERSION = "1.0"


def normalize_feature_names(names: list[str]) -> list[str]:
    return [name.replace(".", "-") for name in names]


def normalize_groups(groups: list | pd.Series) -> list[str]:
    result: list[str] = []
    for val in groups:
        if isinstance(val, float) and math.isnan(val):
            result.append("")
        elif isinstance(val, float) and val == int(val):
            result.append(str(int(val)))
        else:
            result.append(str(val))
    return result


def parse_dataset_name(name: str) -> tuple[str, str]:
    cancer_type, clinical_variable = name.split("_", 1)
    return cancer_type, clinical_variable


def discover_datasets(raw_dir: Path) -> list[str]:
    names = []
    for f in sorted(raw_dir.glob("*_raw.csv")):
        names.append(f.stem.removesuffix("_raw"))
    return names


def read_raw_data(dataset: str, raw_dir: Path) -> dict:
    path = raw_dir / f"{dataset}_raw.csv"
    df = pd.read_csv(path)

    sample_col = (
        "Sample" if "Sample" in df.columns
        else "samples" if "samples" in df.columns
        else None
    )

    sample_ids = df[sample_col].astype(str).tolist() if sample_col else None

    groups = normalize_groups(df["groups"])

    meta_cols = {"groups"}
    if sample_col:
        meta_cols.add(sample_col)
    feature_cols = [c for c in df.columns if c not in meta_cols]

    feature_names = normalize_feature_names(feature_cols)
    expression = df[feature_cols].values.astype(np.float64)

    return {
        "expression": expression,
        "groups": groups,
        "sample_ids": sample_ids,
        "feature_names": feature_names,
    }


def read_processed_data(
    dataset: str, processed_dir: Path, raw_feature_names: list[str]
) -> dict[str, dict]:
    raw_set = set(raw_feature_names)
    result: dict[str, dict] = {}

    for norm, suffix in NORM_SUFFIXES.items():
        path = processed_dir / f"{dataset}{suffix}"
        df = pd.read_csv(path)

        meta_cols = {"groups"}
        for col in ("Sample", "samples"):
            if col in df.columns:
                meta_cols.add(col)
        feature_cols = [c for c in df.columns if c not in meta_cols]

        feature_names = normalize_feature_names(feature_cols)

        missing = set(feature_names) - raw_set
        if missing:
            raise ValueError(
                f"{dataset} {norm}: {len(missing)} feature(s) not found in raw: "
                f"{sorted(missing)[:5]}"
            )

        groups = normalize_groups(df["groups"])
        expression = df[feature_cols].values.astype(np.float64)

        result[norm] = {
            "expression": expression,
            "groups": groups,
            "feature_names": feature_names,
        }

    return result


def read_synthetic_data(
    dataset: str,
    synthetic_dir: Path,
    processed_features: dict[str, list[str]],
) -> dict[str, dict[str, dict]]:
    h5_path = synthetic_dir / dataset / f"{dataset}_generated.h5"
    result: dict[str, dict[str, dict]] = {}

    for src_norm, out_norm in SYNTH_NORM_MAP.items():
        result[out_norm] = {}
        expected_features = processed_features[out_norm]

        for model in MODELS:
            key = f"/{src_norm}/{model}"
            df = pd.read_hdf(h5_path, key=key)

            feature_cols = [c for c in df.columns if c != "groups"]
            feature_names = normalize_feature_names(feature_cols)

            if feature_names != expected_features:
                raise ValueError(
                    f"{dataset} synthetic {out_norm}/{model}: "
                    f"feature names do not match processed features"
                )

            groups = normalize_groups(df["groups"])
            expression = df[feature_cols].values.astype(np.float64)

            result[out_norm][model] = {
                "expression": expression,
                "groups": groups,
                "feature_names": feature_names,
            }

    return result


def read_epochs_trained(
    dataset: str, run_summary_path: Path
) -> dict[tuple[str, str], int]:
    rs = pd.read_csv(run_summary_path)
    ds_rows = rs[rs["dataset"] == dataset]
    result: dict[tuple[str, str], int] = {}

    for _, row in ds_rows.iterrows():
        norm = SYNTH_NORM_MAP.get(row["norm"], row["norm"])
        model = MODEL_SUMMARY_TO_KEY.get(row["model"], row["model"])
        result[(norm, model)] = int(row["epochs_trained"])

    return result


def _get_syng_version() -> str:
    try:
        import syng_bts
        return syng_bts.__version__
    except ImportError:
        return "unknown"


def _write_string_dataset(group: h5py.Group, name: str, data: list[str]) -> None:
    group.create_dataset(name, data=data, dtype=h5py.string_dtype())


def _write_expression_dataset(
    group: h5py.Group, name: str, data: np.ndarray
) -> None:
    group.create_dataset(
        name, data=data, dtype=np.float64, compression="gzip", compression_opts=4
    )


def write_hdf5(
    out_path: Path,
    dataset: str,
    raw_data: dict,
    processed_data: dict[str, dict],
    synthetic_data: dict[str, dict[str, dict]],
    epochs: dict[tuple[str, str], int],
) -> None:
    cancer_type, clinical_variable = parse_dataset_name(dataset)

    all_groups: set[str] = set()
    for g in raw_data["groups"]:
        if g:
            all_groups.add(g)
    for norm_data in processed_data.values():
        all_groups.update(norm_data["groups"])
    group_labels = sorted(all_groups)

    n_raw = raw_data["expression"].shape[0]
    n_raw_feat = raw_data["expression"].shape[1]
    n_filt = processed_data["raw_norm"]["expression"].shape[0]
    n_filt_feat = processed_data["raw_norm"]["expression"].shape[1]

    with h5py.File(out_path, "w") as f:
        # Root attributes
        f.attrs["dataset_name"] = dataset
        f.attrs["cancer_type"] = cancer_type
        f.attrs["clinical_variable"] = clinical_variable
        f.attrs["group_labels"] = group_labels
        f.attrs["version"] = HDF5_VERSION
        f.attrs["creation_date"] = datetime.now(timezone.utc).isoformat()
        f.attrs["syng_bts_version"] = _get_syng_version()
        f.attrs["n_raw_samples"] = n_raw
        f.attrs["n_filtered_samples"] = n_filt
        f.attrs["n_raw_features"] = n_raw_feat
        f.attrs["n_filtered_features"] = n_filt_feat

        # /raw/
        raw_grp = f.create_group("raw")
        _write_expression_dataset(raw_grp, "expression", raw_data["expression"])
        _write_string_dataset(raw_grp, "groups", raw_data["groups"])
        if raw_data["sample_ids"] is not None:
            _write_string_dataset(raw_grp, "sample_ids", raw_data["sample_ids"])
        _write_string_dataset(raw_grp, "feature_names", raw_data["feature_names"])

        # /processed/{norm}/
        proc_grp = f.create_group("processed")
        for norm in NORMS:
            norm_grp = proc_grp.create_group(norm)
            for attr_k, attr_v in NORM_ATTRS[norm].items():
                norm_grp.attrs[attr_k] = attr_v
            nd = processed_data[norm]
            _write_expression_dataset(norm_grp, "expression", nd["expression"])
            _write_string_dataset(norm_grp, "groups", nd["groups"])
            _write_string_dataset(norm_grp, "feature_names", nd["feature_names"])

        # /synthetic/
        synth_grp = f.create_group("synthetic")
        for attr_k, attr_v in SYNTH_SHARED_ATTRS.items():
            synth_grp.attrs[attr_k] = attr_v

        for norm in NORMS:
            norm_grp = synth_grp.create_group(norm)
            _write_string_dataset(
                norm_grp,
                "feature_names",
                processed_data[norm]["feature_names"],
            )
            for model in MODELS:
                model_grp = norm_grp.create_group(model)
                params = MODEL_PARAMS[model]
                model_grp.attrs["kl_weight"] = params["kl_weight"]
                model_grp.attrs["reconstruction_term_weight"] = params[
                    "reconstruction_term_weight"
                ]
                model_grp.attrs["epochs_trained"] = epochs[(norm, model)]
                model_grp.attrs["normalization"] = norm

                sd = synthetic_data[norm][model]
                _write_expression_dataset(
                    model_grp, "expression", sd["expression"]
                )
                _write_string_dataset(model_grp, "groups", sd["groups"])


def _read_h5_strings(dataset: h5py.Dataset) -> list[str]:
    raw = dataset[:]
    return [x.decode() if isinstance(x, bytes) else x for x in raw]


def validate_hdf5(
    path: Path,
    raw_data: dict,
    processed_data: dict[str, dict],
    synthetic_data: dict[str, dict[str, dict]],
) -> list[str]:
    errors: list[str] = []

    with h5py.File(path, "r") as f:
        # Root attributes
        for attr in [
            "dataset_name", "cancer_type", "clinical_variable", "group_labels",
            "version", "creation_date", "syng_bts_version",
            "n_raw_samples", "n_filtered_samples", "n_raw_features",
            "n_filtered_features",
        ]:
            if attr not in f.attrs:
                errors.append(f"Missing root attribute: {attr}")

        # /raw/
        if "raw" not in f:
            errors.append("Missing group: raw")
        else:
            raw_grp = f["raw"]
            expected_shape = raw_data["expression"].shape
            if "expression" not in raw_grp:
                errors.append("Missing dataset: raw/expression")
            elif raw_grp["expression"].shape != expected_shape:
                errors.append(
                    f"raw/expression shape {raw_grp['expression'].shape} "
                    f"!= expected {expected_shape}"
                )
            if "feature_names" in raw_grp:
                stored = _read_h5_strings(raw_grp["feature_names"])
                if stored != raw_data["feature_names"]:
                    errors.append("raw/feature_names mismatch")

        # /processed/{norm}/
        for norm in NORMS:
            grp_path = f"processed/{norm}"
            if grp_path not in f:
                errors.append(f"Missing group: {grp_path}")
                continue
            grp = f[grp_path]
            expected_shape = processed_data[norm]["expression"].shape
            if "expression" not in grp:
                errors.append(f"Missing dataset: {grp_path}/expression")
            elif grp["expression"].shape != expected_shape:
                errors.append(
                    f"{grp_path}/expression shape {grp['expression'].shape} "
                    f"!= expected {expected_shape}"
                )
            if "feature_names" in grp:
                stored = _read_h5_strings(grp["feature_names"])
                if stored != processed_data[norm]["feature_names"]:
                    errors.append(f"{grp_path}/feature_names mismatch")

        # group_labels consistency
        if "group_labels" in f.attrs and "processed" in f:
            stored_labels = sorted(str(x) for x in f.attrs["group_labels"])
            actual_labels: set[str] = set()
            for norm in NORMS:
                grp_path = f"processed/{norm}"
                if grp_path in f and "groups" in f[grp_path]:
                    for val in _read_h5_strings(f[grp_path]["groups"]):
                        if val:
                            actual_labels.add(val)
            if stored_labels != sorted(actual_labels):
                errors.append(
                    f"group_labels attr {stored_labels} != "
                    f"labels found in data {sorted(actual_labels)}"
                )

        # /synthetic/{norm}/{model}/
        for norm in NORMS:
            for model in MODELS:
                grp_path = f"synthetic/{norm}/{model}"
                if grp_path not in f:
                    errors.append(f"Missing group: {grp_path}")
                    continue
                grp = f[grp_path]
                expected_shape = synthetic_data[norm][model]["expression"].shape
                if "expression" not in grp:
                    errors.append(f"Missing dataset: {grp_path}/expression")
                elif grp["expression"].shape != expected_shape:
                    errors.append(
                        f"{grp_path}/expression shape {grp['expression'].shape} "
                        f"!= expected {expected_shape}"
                    )

    return errors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(output_dir: Path) -> None:
    datasets = []
    for h5_path in sorted(output_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            entry = {
                "dataset_name": str(f.attrs["dataset_name"]),
                "cancer_type": str(f.attrs["cancer_type"]),
                "clinical_variable": str(f.attrs["clinical_variable"]),
                "group_labels": list(f.attrs["group_labels"]),
                "n_raw_samples": int(f.attrs["n_raw_samples"]),
                "n_filtered_samples": int(f.attrs["n_filtered_samples"]),
                "n_raw_features": int(f.attrs["n_raw_features"]),
                "n_filtered_features": int(f.attrs["n_filtered_features"]),
                "file": h5_path.name,
                "file_size_bytes": h5_path.stat().st_size,
                "sha256": _sha256(h5_path),
            }
            datasets.append(entry)

    manifest = {
        "version": HDF5_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "syng_bts_version": _get_syng_version(),
        "datasets": datasets,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Manifest written: %s (%d datasets)", manifest_path, len(datasets))


def process_dataset(
    dataset: str,
    *,
    raw_dir: Path,
    processed_dir: Path,
    synthetic_dir: Path,
    run_summary_path: Path,
    output_dir: Path,
    force: bool,
) -> bool:
    out_path = output_dir / f"{dataset}.h5"

    if out_path.exists() and not force:
        log.info("Skipping %s (output exists, use --force to overwrite)", dataset)
        return True

    log.info("Processing %s ...", dataset)

    raw_data = read_raw_data(dataset, raw_dir)
    processed_data = read_processed_data(
        dataset, processed_dir, raw_data["feature_names"]
    )

    processed_features = {
        norm: data["feature_names"] for norm, data in processed_data.items()
    }
    synthetic_data = read_synthetic_data(dataset, synthetic_dir, processed_features)
    epochs = read_epochs_trained(dataset, run_summary_path)

    write_hdf5(out_path, dataset, raw_data, processed_data, synthetic_data, epochs)

    errors = validate_hdf5(out_path, raw_data, processed_data, synthetic_data)
    if errors:
        log.error("%s validation failed:\n  %s", dataset, "\n  ".join(errors))
        out_path.unlink(missing_ok=True)
        return False

    log.info("%s done (%s)", dataset, f"{out_path.stat().st_size / 1e6:.1f} MB")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate TCGA miRNA datasets into HDF5 files."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process even if output exists"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    run_summary_path = SYNTHETIC_DIR / "run_summary.csv"

    datasets = discover_datasets(RAW_DIR)
    log.info("Found %d datasets", len(datasets))

    succeeded = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for dataset in datasets:
        try:
            ok = process_dataset(
                dataset,
                raw_dir=RAW_DIR,
                processed_dir=PROCESSED_DIR,
                synthetic_dir=SYNTHETIC_DIR,
                run_summary_path=run_summary_path,
                output_dir=OUTPUT_DIR,
                force=args.force,
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
                failures.append((dataset, "validation failed"))
        except Exception as exc:
            log.error("%s failed: %s", dataset, exc)
            failed += 1
            failures.append((dataset, str(exc)))
            partial = OUTPUT_DIR / f"{dataset}.h5"
            partial.unlink(missing_ok=True)

    write_manifest(OUTPUT_DIR)

    log.info("Summary: %d succeeded, %d failed", succeeded, failed)
    if failures:
        for name, reason in failures:
            log.error("  FAILED: %s — %s", name, reason)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
