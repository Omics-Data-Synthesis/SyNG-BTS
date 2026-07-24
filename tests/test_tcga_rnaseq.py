"""Tests for the bulk RNA-seq (schema 2.0) path of syng_bts.tcga."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from syng_bts import tcga
from syng_bts.tcga import list_tcga_datasets, load_tcga_dataset

NORMS = ("raw_norm", "TC", "DESeq")
OFFAUGS = ("none", "AE_head")
KL_WEIGHTS = (50, 100, 200)

FIXTURE_BASE_URL = "https://fixture.test/data-rnaseq-v1.0"
FIXTURE_MANIFEST_URL = f"{FIXTURE_BASE_URL}/manifest.json"


def _dataset_url(file: str) -> str:
    return f"{FIXTURE_BASE_URL}/{file}"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_test_rnaseq_h5(
    path: Path,
    *,
    dataset_name: str,
    model_family: str = "CVAE",
    n_train: int = 6,
    n_test: int = 3,
    n_features: int = 5,
    new_size: int = 8,
    group_labels: tuple[str, ...] = ("0", "1"),
    rng_seed: int = 0,
) -> dict:
    """Create a tiny but schema-valid v2.0 HDF5 file at ``path``.

    ``model_family="VAE"`` omits every synthetic ``groups`` dataset, matching
    the unconditional cohorts (LAML, PAAD, READ).
    """
    rng = np.random.default_rng(rng_seed)
    n_samples = n_train + n_test
    has_groups = model_family == "CVAE"
    models = tuple(f"{model_family}1-{kl}" for kl in KL_WEIGHTS)

    features = [f"ENSG{i:011d}.{i % 20 + 1}" for i in range(n_features)]
    sample_ids = [f"TCGA.XX.{i:04d}.01A" for i in range(n_samples)]
    split = ["train"] * n_train + ["test"] * n_test
    groups = [group_labels[i % len(group_labels)] for i in range(n_samples)]

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["cancer_type"] = dataset_name
        f.attrs["clinical_variable"] = ""
        f.attrs["source_label"] = f"{dataset_name}_5-2"
        f.attrs["modality"] = "rnaseq"
        f.attrs["version"] = "2.0"
        f.attrs["group_labels"] = list(group_labels)
        f.attrs["feature_id_type"] = "ensembl_gene_id_versioned"
        f.attrs["model_family"] = model_family
        f.attrs["synthetic_has_groups"] = has_groups
        f.attrs["n_samples"] = n_samples
        f.attrs["n_train"] = n_train
        f.attrs["n_test"] = n_test
        f.attrs["n_features"] = n_features
        f.attrs["creation_date"] = "2026-07-24T00:00:00+00:00"
        f.attrs["syng_bts_version"] = "3.5.0"

        proc = f.create_group("processed")
        for norm in NORMS:
            g = proc.create_group(norm)
            g.attrs["normalization_method"] = (
                "raw" if norm == "raw_norm" else norm
            )
            g.attrs["transform"] = "log2(x+1)"
            g.attrs["scale"] = "log2"
            g.attrs["n_train"] = n_train
            g.attrs["n_test"] = n_test
            g.create_dataset(
                "expression",
                data=rng.random((n_samples, n_features)),
                dtype=np.float32,
            )
            g.create_dataset("groups", data=groups, dtype=h5py.string_dtype())
            g.create_dataset(
                "sample_ids", data=sample_ids, dtype=h5py.string_dtype()
            )
            g.create_dataset("split", data=split, dtype=h5py.string_dtype())
            g.create_dataset(
                "feature_names", data=features, dtype=h5py.string_dtype()
            )

        synth = f.create_group("synthetic")
        synth.attrs["num_epochs"] = 10000
        synth.attrs["seed"] = 123
        synth.attrs["latent_size"] = 32
        synth.attrs["apply_log"] = True
        synth.attrs["reconstruction_term_weight"] = 1

        for norm in NORMS:
            ng = synth.create_group(norm)
            ng.create_dataset(
                "feature_names", data=features, dtype=h5py.string_dtype()
            )
            for offaug in OFFAUGS:
                og = ng.create_group(offaug)
                og.attrs["off_aug"] = offaug
                if offaug == "AE_head":
                    og.attrs["AE_head_num"] = 2
                for model, kl in zip(models, KL_WEIGHTS, strict=True):
                    mg = og.create_group(model)
                    mg.attrs["modelname"] = model_family
                    mg.attrs["kl_weight"] = kl
                    mg.attrs["epochs_trained"] = 100 + kl
                    mg.attrs["early_stop_patience"] = 200
                    mg.attrs["new_size"] = new_size
                    mg.attrs["normalization"] = norm
                    mg.create_dataset(
                        "expression",
                        data=rng.random((new_size, n_features)),
                        dtype=np.float32,
                    )
                    if has_groups:
                        mg.create_dataset(
                            "groups",
                            data=[
                                group_labels[i % len(group_labels)]
                                for i in range(new_size)
                            ],
                            dtype=h5py.string_dtype(),
                        )

    return {
        "dataset_name": dataset_name,
        "cancer_type": dataset_name,
        "clinical_variable": "",
        "source_label": f"{dataset_name}_5-2",
        "group_labels": list(group_labels),
        "n_samples": n_samples,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "normalizations": list(NORMS),
        "off_augs": list(OFFAUGS),
        "models": list(models),
        "model_family": model_family,
        "synthetic_has_groups": has_groups,
        "new_size": new_size,
        "file": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256_of_file(path),
    }


def make_test_rnaseq_manifest(*entries: dict, version: str = "1.0") -> dict:
    return {
        "version": version,
        "modality": "rnaseq",
        "schema_version": "2.0",
        "created": "2026-07-24T00:00:00+00:00",
        "syng_bts_version": "3.5.0",
        "datasets": list(entries),
    }


class TestFixtureBuilder:
    def test_cvae_cohort_has_synthetic_groups(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        entry = make_test_rnaseq_h5(path, dataset_name="SKCM")
        assert entry["synthetic_has_groups"] is True
        assert entry["models"] == ["CVAE1-50", "CVAE1-100", "CVAE1-200"]
        with h5py.File(path, "r") as f:
            assert f.attrs["version"] == "2.0"
            assert "groups" in f["synthetic/DESeq/none/CVAE1-50"]
            assert f["processed/DESeq/expression"].shape == (9, 5)

    def test_vae_cohort_omits_synthetic_groups(self, tmp_path):
        path = tmp_path / "LAML.h5"
        entry = make_test_rnaseq_h5(path, dataset_name="LAML", model_family="VAE")
        assert entry["synthetic_has_groups"] is False
        assert entry["models"] == ["VAE1-50", "VAE1-100", "VAE1-200"]
        with h5py.File(path, "r") as f:
            assert "groups" not in f["synthetic/DESeq/none/VAE1-50"]
            assert "groups" in f["processed/DESeq"]


class TestBuildRnaseq:
    def test_root_attributes(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)

        assert ds.name == "SKCM"
        assert ds.cancer_type == "SKCM"
        assert ds.modality == "rnaseq"
        assert ds.schema_version == "2.0"
        assert ds.source_label == "SKCM_5-2"
        assert ds.clinical_variable == ""
        assert ds.n_samples == 9
        assert ds.n_train == 6
        assert ds.n_test == 3
        assert ds.n_features == 5
        assert ds.model_family == "CVAE"
        assert ds.synthetic_has_groups is True

    def test_raw_is_none(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)
        assert ds.raw is None

    def test_v1_only_counts_are_none(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)
        assert ds.n_raw_samples is None
        assert ds.n_filtered_samples is None
        assert ds.n_raw_features is None
        assert ds.n_filtered_features is None

    def test_processed_carries_sample_ids_and_split(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)

        sub = ds.processed["DESeq"]
        assert sub.expression.index.name == "sample_id"
        assert sub.expression.index[0] == "TCGA.XX.0000.01A"
        assert sub.metadata["split"] == ["train"] * 6 + ["test"] * 3
        assert sub.metadata["n_train"] == 6
        assert sub.metadata["n_test"] == 3

    def test_synthetic_is_three_levels_deep(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)

        assert set(ds.synthetic) == {"raw_norm", "TC", "DESeq"}
        assert set(ds.synthetic["DESeq"]) == {"none", "AE_head"}
        assert set(ds.synthetic["DESeq"]["none"]) == {
            "CVAE1-50",
            "CVAE1-100",
            "CVAE1-200",
        }
        sub = ds.synthetic["DESeq"]["none"]["CVAE1-50"]
        assert sub.expression.shape == (8, 5)

    def test_synthetic_inherits_shared_attrs(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)

        md = ds.synthetic["DESeq"]["none"]["CVAE1-50"].metadata
        assert md["seed"] == 123
        assert md["latent_size"] == 32
        assert md["kl_weight"] == 50
        assert md["epochs_trained"] == 150
        assert md["off_aug"] == "none"

    def test_ae_head_group_carries_head_num(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        ds = tcga._build_dataset_from_h5(path)
        md = ds.synthetic["DESeq"]["AE_head"]["CVAE1-50"].metadata
        assert md["off_aug"] == "AE_head"
        assert md["AE_head_num"] == 2

    def test_vae_cohort_synthetic_groups_are_none(self, tmp_path):
        path = tmp_path / "LAML.h5"
        make_test_rnaseq_h5(path, dataset_name="LAML", model_family="VAE")
        ds = tcga._build_dataset_from_h5(path)

        assert ds.synthetic_has_groups is False
        assert ds.synthetic["DESeq"]["none"]["VAE1-50"].groups is None
        assert ds.processed["DESeq"].groups is not None


class TestAccessors:
    @pytest.fixture
    def cvae_ds(self, tmp_path):
        path = tmp_path / "SKCM.h5"
        make_test_rnaseq_h5(path, dataset_name="SKCM")
        return tcga._build_dataset_from_h5(path)

    @pytest.fixture
    def vae_ds(self, tmp_path):
        path = tmp_path / "LAML.h5"
        make_test_rnaseq_h5(path, dataset_name="LAML", model_family="VAE")
        return tcga._build_dataset_from_h5(path)

    def test_real_returns_full_cohort(self, cvae_ds):
        expr, groups = cvae_ds.real("DESeq")
        assert expr.shape == (9, 5)
        assert len(groups) == 9

    def test_real_rejects_bad_normalization(self, cvae_ds):
        with pytest.raises(ValueError, match="Invalid normalization"):
            cvae_ds.real("quantile")

    def test_synth_default_model_is_lowest_kl(self, cvae_ds):
        expr, _ = cvae_ds.synth("DESeq")
        expected, _ = cvae_ds.synth("DESeq", "CVAE1-50")
        assert expr.equals(expected)

    def test_synth_default_model_is_lowest_kl_for_vae(self, vae_ds):
        expr, _ = vae_ds.synth("DESeq")
        expected, _ = vae_ds.synth("DESeq", "VAE1-50")
        assert expr.equals(expected)

    @pytest.mark.parametrize("off_aug", [None, "none"])
    def test_off_aug_none_and_string_are_equivalent(self, cvae_ds, off_aug):
        expr, _ = cvae_ds.synth("DESeq", "CVAE1-50", off_aug=off_aug)
        assert expr.shape == (8, 5)

    def test_off_aug_ae_head(self, cvae_ds):
        expr, _ = cvae_ds.synth("DESeq", "CVAE1-50", off_aug="AE_head")
        plain, _ = cvae_ds.synth("DESeq", "CVAE1-50", off_aug="none")
        assert expr.shape == (8, 5)
        assert not expr.equals(plain)

    def test_unknown_off_aug_raises(self, cvae_ds):
        with pytest.raises(ValueError, match="Invalid off_aug"):
            cvae_ds.synth("DESeq", "CVAE1-50", off_aug="Gaussian_head")

    def test_wrong_model_family_error_lists_real_options(self, vae_ds):
        with pytest.raises(ValueError, match="VAE1-50"):
            vae_ds.synth("DESeq", "CVAE1-50")

    def test_vae_synth_groups_is_none(self, vae_ds):
        expr, groups = vae_ds.synth("DESeq", "VAE1-50")
        assert groups is None
        assert expr.shape == (8, 5)

    def test_repr_shows_rnaseq_models_not_mirna(self, vae_ds):
        text = repr(vae_ds)
        assert "VAE1-50" in text
        assert "CVAE1_5" not in text
        assert "LAML" in text

    def test_repr_does_not_show_none_counts(self, cvae_ds):
        text = repr(cvae_ds)
        assert "None samples" not in text


class TestOffAugOnMirna:
    def test_off_aug_on_mirna_raises(self, tmp_path):
        import tests.test_tcga as t1

        path = tmp_path / "BRCA_carcinoma.h5"
        t1.make_test_h5(path, dataset_name="BRCA_carcinoma")
        ds = tcga._build_dataset_from_h5(path)

        with pytest.raises(ValueError, match="modality"):
            ds.synth("DESeq", "CVAE1_5", off_aug="AE_head")


class TestLoadRnaseqDataset:
    @pytest.fixture
    def served(self, monkeypatch, network_stub, cache_root, tmp_path):
        h5_dir = tmp_path / "_fixture"
        h5_dir.mkdir()
        skcm = make_test_rnaseq_h5(h5_dir / "SKCM.h5", dataset_name="SKCM")
        laml = make_test_rnaseq_h5(
            h5_dir / "LAML.h5", dataset_name="LAML", model_family="VAE"
        )
        manifest = make_test_rnaseq_manifest(skcm, laml)
        network_stub.serve(FIXTURE_MANIFEST_URL, json.dumps(manifest).encode())
        for name in ("SKCM", "LAML"):
            network_stub.serve(
                _dataset_url(f"{name}.h5"), (h5_dir / f"{name}.h5").read_bytes()
            )
        monkeypatch.setattr(
            tcga, "_DEFAULT_MANIFEST_URL", {"rnaseq": FIXTURE_MANIFEST_URL}
        )
        return manifest

    def test_list_returns_bare_codes(self, served):
        assert list_tcga_datasets(modality="rnaseq") == ["LAML", "SKCM"]
        assert list_tcga_datasets(modality="rnaseq", short=True) == ["LAML", "SKCM"]

    def test_load_caches_under_modality_dir(self, served, cache_root):
        ds = load_tcga_dataset("SKCM", modality="rnaseq")
        assert ds.modality == "rnaseq"
        assert (cache_root / "tcga" / "rnaseq" / "1.0" / "SKCM.h5").exists()

    def test_same_code_resolves_differently_per_modality(
        self, served, cache_root, monkeypatch, network_stub, tmp_path
    ):
        """SKCM exists in both bundles and must resolve to different files."""
        import tests.test_tcga as t1

        # Serve a miRNA bundle whose SKCM entry has a different full name.
        mirna_dir = tmp_path / "_mirna"
        mirna_dir.mkdir()
        mirna_path = mirna_dir / "SKCM_initial_pathologic_dx_year.h5"
        mirna_entry = t1.make_test_h5(
            mirna_path, dataset_name="SKCM_initial_pathologic_dx_year"
        )
        mirna_manifest = t1.make_test_manifest(mirna_entry)
        mirna_url = "https://fixture.test/data-v1.0/manifest.json"
        network_stub.serve(mirna_url, json.dumps(mirna_manifest).encode())
        network_stub.serve(
            "https://fixture.test/data-v1.0/SKCM_initial_pathologic_dx_year.h5",
            mirna_path.read_bytes(),
        )
        monkeypatch.setattr(
            tcga,
            "_DEFAULT_MANIFEST_URL",
            {"mirna": mirna_url, "rnaseq": FIXTURE_MANIFEST_URL},
        )

        bulk = load_tcga_dataset("SKCM", modality="rnaseq")
        mirna = load_tcga_dataset("SKCM", modality="mirna")

        assert bulk.name == "SKCM"
        assert mirna.name == "SKCM_initial_pathologic_dx_year"
        assert bulk.modality == "rnaseq"
        assert mirna.modality == "mirna"
        assert (cache_root / "tcga" / "rnaseq" / "1.0" / "SKCM.h5").exists()
        assert (
            cache_root
            / "tcga"
            / "mirna"
            / "1.0"
            / "SKCM_initial_pathologic_dx_year.h5"
        ).exists()
