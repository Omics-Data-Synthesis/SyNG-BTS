"""
Use Python SyntheSize tooling (get_data_metrics + visualize)
to draw UMAP and heatmap for Augmentation-FiveSubtypes-2026-02-24.

Run from repo root, e.g.:

    cd /Users/yanjiechen/Documents/Github/SyNG-BTS_2.6
    python evaluations/visualize_UMAP_FiveSubtypes_2026_03_12.py

Tune CANCER / NORM / MODEL / BATCH / DATA_TYPE below as needed.
"""

import os
import sys
import csv
import tempfile
import traceback

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Import SyntheSize helper functions (same style as notebook)
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # evaluations/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)               # repo root
sys.path.insert(0, PROJECT_ROOT)

try:
    from syng_bts.tools import get_data_metrics, visualize
except ImportError:
    # Fallback: direct import by file location
    tools_path = os.path.join(PROJECT_ROOT, "syng_bts", "tools.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("tools", tools_path)
    tools_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_module)
    get_data_metrics = tools_module.get_data_metrics
    visualize = tools_module.visualize


# ---------------------------------------------------------------------
# Parameters: which configuration to visualize
# ---------------------------------------------------------------------

CANCER = "COAD"          # "COAD", "LAML", "PAAD", "READ", "SKCM"
NORM = "DESeq"             # "raw", "TC", "DESeq"
MODEL = "CVAE1-100"       # e.g. "CVAE1-10", "CVAE1-50", "CVAE1-100"
BATCH = 1                # integer
DATA_TYPE = "miRNA"      # currently "miRNA"; extend if needed


# ---------------------------------------------------------------------
# Construct paths for real (test) and generated files
# ---------------------------------------------------------------------

ROOT_AUG = os.path.join(
    PROJECT_ROOT,
    "syng_bts",
    "data",
    "case",
    "Augmentation-FiveSubtypes-2026-02-24",
)
cancer_prefix = f"{CANCER}_5-2"

batch_dir = os.path.join(
    ROOT_AUG,
    cancer_prefix,
    NORM,
    f"batch_{BATCH}",
)

real_file_name = os.path.join(
    batch_dir,
    f"{cancer_prefix}_{NORM}_batch_{BATCH}_test.csv",
)

generated_file_name = os.path.join(
    batch_dir,
    MODEL,
    DATA_TYPE,
    f"{cancer_prefix}_{NORM}_batch_{BATCH}_train_epochES_batch01_{MODEL}_generated.csv",
)

print("Real file     :", real_file_name)
print("Generated file:", generated_file_name)

if not os.path.exists(real_file_name):
    raise FileNotFoundError(f"Real file not found: {real_file_name}")
if not os.path.exists(generated_file_name):
    raise FileNotFoundError(f"Generated file not found: {generated_file_name}")


# ---------------------------------------------------------------------
# Pre-processing of generated file
# (follow the same style as Experiments_run_SyntheSize_robust.ipynb)
# ---------------------------------------------------------------------

temp_files_to_cleanup = []


def remove_all_zero_second_last_column(path: str) -> str:
    """If the second-to-last column is all zeros, remove it into a temp file."""
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows or len(rows[0]) <= 1:
            return path

        second_to_last_idx = -2
        vals = []
        for row in rows:
            if len(row) > abs(second_to_last_idx):
                v = row[second_to_last_idx]
                if not isinstance(v, str):
                    v = str(v)
                vals.append(v.strip())

        def is_zero_value(s: str) -> bool:
            try:
                return abs(float(s)) < 1e-10
            except (ValueError, TypeError):
                return s in ("", "0", "0.0")

        is_all_zero = bool(vals) and all(is_zero_value(s) for s in vals)
        if not is_all_zero:
            print("Second-to-last column is not all zeros; keep original generated file.")
            return path

        print("Second-to-last column is all zeros; creating temp file without that column...")
        fd, tmp_path = tempfile.mkstemp(
            suffix=".csv",
            prefix="generated_temp_",
            dir=os.path.dirname(path) or ".",
        )
        os.close(fd)

        with open(tmp_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            for row in rows:
                if len(row) > 1:
                    new_row = row[:second_to_last_idx] + row[second_to_last_idx + 1 :]
                    writer.writerow(new_row)
                else:
                    writer.writerow(row)

        temp_files_to_cleanup.append(tmp_path)
        print("Temp (drop col) generated file:", tmp_path)
        return tmp_path
    except Exception as e:
        print("Error checking/removing all-zero second-last column:", e)
        traceback.print_exc()
        return path


def swap_last_two_columns_if_needed(path: str) -> str:
    """
    If header's last two columns are 'samples, groups', swap to 'groups, samples'
    and return a temp file path; otherwise return original path.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows or len(rows[0]) < 2:
            return path

        header = rows[0]
        last_two = header[-2:]
        if last_two[0].strip() == "samples" and last_two[1].strip() == "groups":
            print(f"Detected last two columns 'samples, groups' in {path}, swapping...")
            fd, tmp_path = tempfile.mkstemp(
                suffix=".csv",
                prefix="swap_cols_temp_",
                dir=os.path.dirname(path) or ".",
            )
            os.close(fd)
            with open(tmp_path, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                for row in rows:
                    if len(row) >= 2:
                        new_row = row[:-2] + [row[-1], row[-2]]
                        writer.writerow(new_row)
                    else:
                        writer.writerow(row)
            temp_files_to_cleanup.append(tmp_path)
            print("Temp (swap cols) file:", tmp_path)
            return tmp_path
        else:
            print(f"Last two columns of {path} are not 'samples, groups'; keep as is.")
            return path
    except Exception as e:
        print("Error swapping last two columns if needed:", e)
        traceback.print_exc()
        return path


# 1) drop all-zero second-last column if necessary
generated_file_name = remove_all_zero_second_last_column(generated_file_name)

# 2) ensure last two columns are ordered correctly for both files
real_file_name = swap_last_two_columns_if_needed(real_file_name)
generated_file_name = swap_last_two_columns_if_needed(generated_file_name)


# 3) drop header row of generated, to match get_data_metrics expectation (header=None)
try:
    df_gen = pd.read_csv(generated_file_name, header=0)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".csv",
        prefix="generated_skip_header_",
        dir=os.path.dirname(generated_file_name) or ".",
    )
    os.close(fd)
    df_gen.to_csv(tmp_path, index=False, header=False)
    temp_files_to_cleanup.append(tmp_path)
    generated_file_name = tmp_path
    print("Generated file with header removed:", generated_file_name)
except Exception as e:
    print("Error dropping header of generated file:", e)
    traceback.print_exc()


# ---------------------------------------------------------------------
# Load data using get_data_metrics and call visualize (Python style)
# ---------------------------------------------------------------------

real_data, groups_real, generated_data, groups_generated, unique_types = get_data_metrics(
    real_file_name, generated_file_name
)

# Use same style as notebook; log2(real+1) is done in get_data_metrics; generated is assumed log scale already.

output_dir = os.path.join(PROJECT_ROOT, "evaluations", "UMAP-2026-03-12")
stem = f"{CANCER}_{NORM}_batch{BATCH}_{MODEL}_{DATA_TYPE}"

visualize(
    real_data=real_data,
    groups_real=groups_real,
    unique_types=unique_types,
    generated_data=generated_data,
    groups_generated=groups_generated,
    ratio=1.0,
    seed=88,
    output_dir=output_dir,
    output_heatmap_name=f"{stem}_heatmap.png",
    output_umap_name=f"{stem}_umap.png",
)


# ---------------------------------------------------------------------
# Cleanup temporary files
# ---------------------------------------------------------------------

for tmp in temp_files_to_cleanup:
    try:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
            print("Removed temp file:", tmp)
    except Exception as e:
        print("Error removing temp file:", tmp, e)

