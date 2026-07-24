"""Consolidate the bulk TCGA RNA-seq delivery into one HDF5 file per cohort.

rna-seq data's directory tree:

    {SOURCE}/{COHORT}/{norm}/{stem}_batch_1_{train,test}.csv
    {SOURCE}/{COHORT}/{norm}/{offaug}/{model}/{stem}_..._generated.csv

and writes one schema-2.0 ``{CANCER}.h5`` per cohort plus ``manifest.json``.
Upload the results manually to a GitHub release.

Usage:
    python scripts/prepare_rnaseq_hdf5.py \\
        --source-dir "/path/to/Augmented data for TCGA RNA-seq" \\
        --output-dir full_data/rnaseq_hdf5_output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SOURCE_NORMS = ("raw", "TC", "DESeq")
NORM_KEYS = {"raw": "raw_norm", "TC": "TC", "DESeq": "DESeq"}
META_COLS = {"samples", "groups"}

SCHEMA_VERSION = "2.0"
DATA_VERSION = "1.0"
MODALITY = "rnaseq"

# Constant across all 90 runs; verified in the design spec.
SHARED_SYNTH_ATTR_KEYS = (
    "num_epochs",
    "seed",
    "apply_log",
    "reconstruction_term_weight",
)

_OFFAUG_RE = re.compile(r"^offaug_(?:(none)|(?P<name>.+?)_(?P<num>\d+))$")


def _get_syng_version() -> str:
    try:
        import syng_bts

        return syng_bts.__version__
    except ImportError:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_string_dataset(group: h5py.Group, name: str, data: list[str]) -> None:
    group.create_dataset(name, data=data, dtype=h5py.string_dtype())


def _write_expression_dataset(
    group: h5py.Group, name: str, data: np.ndarray, dtype: type = np.float32
) -> None:
    group.create_dataset(
        name, data=data, dtype=dtype, compression="gzip", compression_opts=4
    )


def normalize_groups(values: list | pd.Series) -> list[str]:
    """Coerce heterogeneous group labels to clean strings ('1.0' -> '1')."""
    result: list[str] = []
    for val in values:
        if isinstance(val, float) and math.isnan(val):
            result.append("")
        elif isinstance(val, float) and val == int(val):
            result.append(str(int(val)))
        else:
            result.append(str(val))
    return result


def _one(paths: list[Path], what: str, where: Path) -> Path:
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one {what} in {where}, found {len(paths)}: "
            f"{[p.name for p in paths]}"
        )
    return paths[0]


def parse_offaug(dirname: str) -> tuple[str, int | None]:
    """'offaug_none' -> ('none', None); 'offaug_AE_head_2' -> ('AE_head', 2)."""
    m = _OFFAUG_RE.match(dirname)
    if not m:
        raise ValueError(f"Unrecognized offaug directory name: {dirname}")
    if m.group(1) == "none":
        return "none", None
    return m.group("name"), int(m.group("num"))


def read_processed(cohort_dir: Path) -> dict:
    """Read train+test for all three normalizations and validate consistency."""
    per_norm: dict[str, dict] = {}
    ref_samples: list[str] | None = None
    ref_features: list[str] | None = None

    for norm in SOURCE_NORMS:
        ndir = cohort_dir / norm
        train_p = _one(sorted(ndir.glob("*_train.csv")), "train CSV", ndir)
        test_p = _one(sorted(ndir.glob("*_test.csv")), "test CSV", ndir)

        train = pd.read_csv(train_p)
        test = pd.read_csv(test_p)

        features = [c for c in train.columns if c not in META_COLS]
        if [c for c in test.columns if c not in META_COLS] != features:
            raise ValueError(f"{norm}: train/test feature mismatch")

        samples = train["samples"].astype(str).tolist() + (
            test["samples"].astype(str).tolist()
        )
        if len(set(samples)) != len(samples):
            raise ValueError(f"{norm}: train and test sample IDs overlap")

        if ref_samples is None:
            ref_samples, ref_features = samples, features
        else:
            if samples != ref_samples:
                raise ValueError(f"{norm}: sample order differs from other norms")
            if features != ref_features:
                raise ValueError(f"{norm}: feature set differs from other norms")

        block = np.vstack(
            [train[features].to_numpy(), test[features].to_numpy()]
        ).astype(np.float64)
        if not np.isfinite(block).all() or block.min() < 0:
            raise ValueError(
                f"{norm}: negative, NaN, or infinite values, log2(x+1) is invalid"
            )

        per_norm[NORM_KEYS[norm]] = {
            "expression": np.log2(block + 1.0).astype(np.float32),
            "groups": normalize_groups(list(train["groups"]) + list(test["groups"])),
            "n_train": len(train),
            "n_test": len(test),
        }

    return {
        "per_norm": per_norm,
        "sample_ids": ref_samples,
        "feature_names": ref_features,
        "split": ["train"] * per_norm["raw_norm"]["n_train"]
        + ["test"] * per_norm["raw_norm"]["n_test"],
    }


def read_synthetic(cohort_dir: Path, feature_names: list[str]) -> dict:
    """Read every {norm}/{offaug}/{model} run for one cohort."""
    result: dict[str, dict[str, dict]] = {}
    shared: dict = {}
    model_family: str | None = None
    has_groups: bool | None = None
    new_size: int | None = None

    for norm in SOURCE_NORMS:
        key = NORM_KEYS[norm]
        result[key] = {}
        for odir in sorted(p for p in (cohort_dir / norm).iterdir() if p.is_dir()):
            offaug, head_num = parse_offaug(odir.name)
            result[key][offaug] = {"_attrs": {"off_aug": offaug}, "models": {}}
            if head_num is not None:
                result[key][offaug]["_attrs"]["AE_head_num"] = head_num

            for mdir in sorted(p for p in odir.iterdir() if p.is_dir()):
                gen_p = _one(
                    sorted(mdir.glob("*_generated.csv")), "generated CSV", mdir
                )
                meta_p = _one(
                    sorted(mdir.glob("*_metadata.json")), "metadata JSON", mdir
                )
                meta = json.loads(meta_p.read_text())

                gen = pd.read_csv(gen_p)
                cols = [c for c in gen.columns if c != "groups"]
                if cols != feature_names:
                    raise ValueError(
                        f"{mdir}: generated feature names differ from processed"
                    )

                block = gen[cols].to_numpy().astype(np.float64)
                if not np.isfinite(block).all() or block.min() < 0:
                    raise ValueError(
                        f"{mdir}: negative, NaN, or infinite values before log2"
                    )

                grp_files = sorted(mdir.glob("*_generated_groups.csv"))
                groups = None
                if grp_files:
                    groups = normalize_groups(
                        pd.read_csv(grp_files[0])["group"].tolist()
                    )
                    if len(groups) != block.shape[0]:
                        raise ValueError(
                            f"{mdir}: generated_groups.csv has {len(groups)} rows, "
                            f"generated expression has {block.shape[0]} rows"
                        )

                if has_groups is None:
                    has_groups = groups is not None
                    model_family = str(meta["modelname"])
                    new_size = len(gen)
                    shared = {k: meta[k] for k in SHARED_SYNTH_ATTR_KEYS}
                    shared["latent_size"] = meta["arch_params"]["latent_size"]
                else:
                    if str(meta["modelname"]) != model_family:
                        raise ValueError(
                            f"{mdir}: modelname {str(meta['modelname'])!r} differs "
                            f"from cohort's first run {model_family!r}"
                        )
                    if (groups is not None) != has_groups:
                        raise ValueError(
                            f"{mdir}: has_groups={groups is not None} differs from "
                            f"cohort's first run has_groups={has_groups}"
                        )
                    run_shared = {k: meta[k] for k in SHARED_SYNTH_ATTR_KEYS}
                    run_shared["latent_size"] = meta["arch_params"]["latent_size"]
                    if run_shared != shared:
                        raise ValueError(
                            f"{mdir}: shared synthesis attrs {run_shared} differ "
                            f"from cohort's first run {shared}"
                        )

                result[key][offaug]["models"][mdir.name] = {
                    "expression": np.log2(block + 1.0).astype(np.float32),
                    "groups": groups,
                    "attrs": {
                        "modelname": str(meta["modelname"]),
                        "kl_weight": int(meta["kl_weight"]),
                        "epochs_trained": int(meta["epochs_trained"]),
                        "early_stop_patience": int(meta["early_stop_patience"]),
                        "new_size": len(gen),
                        "normalization": key,
                    },
                }

    return {
        "data": result,
        "shared_attrs": shared,
        "model_family": model_family,
        "synthetic_has_groups": bool(has_groups),
        "new_size": new_size,
    }


def write_hdf5(
    out_path: Path,
    *,
    cancer_type: str,
    source_label: str,
    processed: dict,
    synthetic: dict,
) -> None:
    per_norm = processed["per_norm"]
    n_train = per_norm["raw_norm"]["n_train"]
    n_test = per_norm["raw_norm"]["n_test"]
    n_samples = n_train + n_test
    n_features = len(processed["feature_names"])
    group_labels = sorted({g for g in per_norm["raw_norm"]["groups"] if g})

    with h5py.File(out_path, "w") as f:
        f.attrs["dataset_name"] = cancer_type
        f.attrs["cancer_type"] = cancer_type
        f.attrs["clinical_variable"] = ""
        f.attrs["source_label"] = source_label
        f.attrs["modality"] = MODALITY
        f.attrs["version"] = SCHEMA_VERSION
        f.attrs["group_labels"] = group_labels
        f.attrs["feature_id_type"] = "ensembl_gene_id_versioned"
        f.attrs["model_family"] = synthetic["model_family"]
        f.attrs["synthetic_has_groups"] = synthetic["synthetic_has_groups"]
        f.attrs["n_samples"] = n_samples
        f.attrs["n_train"] = n_train
        f.attrs["n_test"] = n_test
        f.attrs["n_features"] = n_features
        f.attrs["creation_date"] = datetime.now(timezone.utc).isoformat()
        f.attrs["syng_bts_version"] = _get_syng_version()

        proc = f.create_group("processed")
        for src_norm in SOURCE_NORMS:
            key = NORM_KEYS[src_norm]
            g = proc.create_group(key)
            g.attrs["normalization_method"] = src_norm
            g.attrs["transform"] = "log2(x+1)"
            g.attrs["scale"] = "log2"
            g.attrs["n_train"] = n_train
            g.attrs["n_test"] = n_test
            nd = per_norm[key]
            _write_expression_dataset(g, "expression", nd["expression"])
            _write_string_dataset(g, "groups", nd["groups"])
            _write_string_dataset(g, "sample_ids", processed["sample_ids"])
            _write_string_dataset(g, "split", processed["split"])
            _write_string_dataset(g, "feature_names", processed["feature_names"])

        synth = f.create_group("synthetic")
        for k, v in synthetic["shared_attrs"].items():
            synth.attrs[k] = v

        for src_norm in SOURCE_NORMS:
            key = NORM_KEYS[src_norm]
            ng = synth.create_group(key)
            _write_string_dataset(ng, "feature_names", processed["feature_names"])
            for offaug, payload in synthetic["data"][key].items():
                og = ng.create_group(offaug)
                for ak, av in payload["_attrs"].items():
                    og.attrs[ak] = av
                for model, md in payload["models"].items():
                    mg = og.create_group(model)
                    for ak, av in md["attrs"].items():
                        mg.attrs[ak] = av
                    _write_expression_dataset(mg, "expression", md["expression"])
                    if md["groups"] is not None:
                        _write_string_dataset(mg, "groups", md["groups"])


def validate_hdf5(path: Path, processed: dict, synthetic: dict) -> list[str]:
    errors: list[str] = []
    n_features = len(processed["feature_names"])
    n_samples = len(processed["sample_ids"])

    with h5py.File(path, "r") as f:
        for norm in NORM_KEYS.values():
            gp = f"processed/{norm}"
            if gp not in f:
                errors.append(f"Missing group: {gp}")
                continue
            if f[gp]["expression"].shape != (n_samples, n_features):
                errors.append(f"{gp}/expression shape mismatch")
            stored = [x.decode() for x in f[gp]["feature_names"][:]]
            if stored != processed["feature_names"]:
                errors.append(f"{gp}/feature_names mismatch")

        for norm, offaugs in synthetic["data"].items():
            for offaug, payload in offaugs.items():
                for model, md in payload["models"].items():
                    gp = f"synthetic/{norm}/{offaug}/{model}"
                    if gp not in f:
                        errors.append(f"Missing group: {gp}")
                        continue
                    if f[gp]["expression"].shape != md["expression"].shape:
                        errors.append(f"{gp}/expression shape mismatch")
                    if (md["groups"] is not None) != ("groups" in f[gp]):
                        errors.append(f"{gp}/groups presence mismatch")

    return errors


def process_cohort(cohort_dir: Path, *, output_dir: Path, force: bool) -> bool:
    cancer_type = cohort_dir.name.split("_", 1)[0]
    out_path = output_dir / f"{cancer_type}.h5"

    if out_path.exists() and not force:
        log.info("Skipping %s (output exists, use --force)", cancer_type)
        return True

    log.info("Processing %s ...", cohort_dir.name)
    processed = read_processed(cohort_dir)
    synthetic = read_synthetic(cohort_dir, processed["feature_names"])

    write_hdf5(
        out_path,
        cancer_type=cancer_type,
        source_label=cohort_dir.name,
        processed=processed,
        synthetic=synthetic,
    )

    errors = validate_hdf5(out_path, processed, synthetic)
    if errors:
        log.error("%s validation failed:\n  %s", cancer_type, "\n  ".join(errors))
        out_path.unlink(missing_ok=True)
        return False

    log.info("%s done (%.1f MB)", cancer_type, out_path.stat().st_size / 1e6)
    return True


def write_manifest(output_dir: Path) -> None:
    datasets = []
    for h5_path in sorted(output_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            norms = list(NORM_KEYS.values())
            # h5py iterates links alphabetically, which would emit
            # ["AE_head", "none"]. Pin the order the spec's manifest shows.
            present = {k for k in f["synthetic/DESeq"] if k != "feature_names"}
            offaugs = [k for k in ("none", "AE_head") if k in present]
            if set(offaugs) != present:
                raise ValueError(f"{h5_path.name}: unexpected off_aug groups {present}")
            first = f[f"synthetic/DESeq/{offaugs[0]}"]
            models = sorted(
                first,
                key=lambda m: int(first[m].attrs["kl_weight"]),
            )
            datasets.append(
                {
                    "dataset_name": str(f.attrs["dataset_name"]),
                    "cancer_type": str(f.attrs["cancer_type"]),
                    "clinical_variable": str(f.attrs["clinical_variable"]),
                    "source_label": str(f.attrs["source_label"]),
                    "group_labels": [str(x) for x in f.attrs["group_labels"]],
                    "n_samples": int(f.attrs["n_samples"]),
                    "n_train": int(f.attrs["n_train"]),
                    "n_test": int(f.attrs["n_test"]),
                    "n_features": int(f.attrs["n_features"]),
                    "normalizations": norms,
                    "off_augs": offaugs,
                    "models": models,
                    "model_family": str(f.attrs["model_family"]),
                    "synthetic_has_groups": bool(f.attrs["synthetic_has_groups"]),
                    "new_size": int(first[models[0]].attrs["new_size"]),
                    "file": h5_path.name,
                    "file_size_bytes": h5_path.stat().st_size,
                    "sha256": _sha256(h5_path),
                }
            )

    manifest = {
        "version": DATA_VERSION,
        "modality": MODALITY,
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "syng_bts_version": _get_syng_version(),
        "datasets": datasets,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written (%d datasets)", len(datasets))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate bulk TCGA RNA-seq datasets into HDF5 files."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = sorted(p for p in args.source_dir.iterdir() if p.is_dir())
    log.info("Found %d cohorts", len(cohorts))

    failures: list[str] = []
    for cohort in cohorts:
        try:
            if not process_cohort(cohort, output_dir=args.output_dir, force=args.force):
                failures.append(cohort.name)
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", cohort.name, exc)
            failures.append(cohort.name)
            (args.output_dir / f"{cohort.name.split('_', 1)[0]}.h5").unlink(
                missing_ok=True
            )

    if failures:
        log.error(
            "Skipping manifest: %d cohort(s) failed (%s). Fix them and rerun; "
            "a manifest written now would describe an incomplete release.",
            len(failures),
            ", ".join(failures),
        )
    else:
        write_manifest(args.output_dir)

    log.info("Summary: %d ok, %d failed", len(cohorts) - len(failures), len(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
