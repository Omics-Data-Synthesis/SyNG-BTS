#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAAD-READ-SKCM 增强完整工作流 (SyNG-BTS 2.6)
- 3 种癌症亚类: PAAD, READ, SKCM（与 COAD-LAML 脚本互补）
- 3 种标准化: raw / TC / DESeq
- 3 个 CVAE 模型: CVAE1-50, CVAE1-100, CVAE1-200
- 2 种 off_aug: None / AE_head
- 1 种 data_type: miRNA
- 20 个 batch
总实验数: 3 × 3 × 3 × 2 × 1 × 20 = 1080 次

输出根目录与 COAD-LAML 批次一致，写入
syng_bts/data/case/Augmentation-COAD-LAML-2026-03-30/
（其下按 case_id 分目录，与 COAD_5-2 / LAML_5-2 并列）。
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", "")

from syng_bts.experiments import ApplyExperiment

_LOG_DIR = _REPO_ROOT / "vignettes"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "Complete_workflow_PAAD-READ-SKCM.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CANCER_SUBTYPES = [
    ("PAAD_5-2", "PAAD"),
    ("READ_5-2", "READ"),
    ("SKCM_5-2", "SKCM"),
]
NORMALIZATION_METHODS = ["raw", "TC", "DESeq"]
MODELS = ["CVAE1-50", "CVAE1-100", "CVAE1-200"]
OFF_AUGS = [None, "AE_head"]
DATA_TYPES = ["miRNA"]
N_BATCHES = 20


def get_data_filename(prefix, normalization_method):
    if normalization_method == "raw":
        return f"{prefix}Positive_5-2.csv"
    return f"{prefix}Positive_5-2_{normalization_method}.csv"


def get_off_aug_tag(off_aug):
    return "offaug_none" if off_aug is None else f"offaug_{off_aug}"


def split_data_by_sample_size(data, case_id, normalization_method, batch_num):
    n_samples = len(data)
    if n_samples < 100:
        logger.info(f"{case_id}_{normalization_method}_batch_{batch_num}: n={n_samples}<100，使用全量数据")
        train_data = data.copy()
        test_data = data.copy()
    elif 100 <= n_samples < 200:
        logger.info(f"{case_id}_{normalization_method}_batch_{batch_num}: n={n_samples}, 50%-50% 划分")
        train_data, test_data = train_test_split(
            data,
            test_size=0.5,
            random_state=batch_num,
            stratify=data["groups"] if "groups" in data.columns else None,
        )
    else:
        logger.info(f"{case_id}_{normalization_method}_batch_{batch_num}: n={n_samples}, 80%-20% 划分")
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=batch_num,
            stratify=data["groups"] if "groups" in data.columns else None,
        )
    train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
    return train_data, test_data


def save_split_data(train_data, test_data, batch_dir, base_filename):
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    train_file = batch_dir / f"{base_filename}_train.csv"
    test_file = batch_dir / f"{base_filename}_test.csv"
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    logger.info(f"已保存: {train_file}, {test_file}")


def create_run_output_dir(
    base_path, case_id, normalization_method, batch_num, off_aug, model_name, data_type
):
    off_aug_tag = get_off_aug_tag(off_aug)
    output_path = (
        Path(base_path)
        / case_id
        / normalization_method
        / f"batch_{batch_num}"
        / off_aug_tag
        / model_name
        / data_type
    )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_experiment_record(
    record_file,
    case_id,
    normalization_method,
    batch_num,
    off_aug,
    model_name,
    data_type,
    status,
    start_time,
    end_time,
    total_experiments,
    completed_experiments,
):
    from datetime import datetime

    completion_ratio = (completed_experiments / total_experiments) * 100
    duration = end_time - start_time
    duration_str = str(duration).split(".")[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    off_aug_tag = get_off_aug_tag(off_aug)
    line = (
        f"{case_id}-{normalization_method}-batch{batch_num}-{off_aug_tag}-{model_name}-{data_type}-{status}"
        f"|{timestamp}|{duration_str}|{completion_ratio:.1f}%|{completed_experiments}/{total_experiments}\n"
    )
    with open(record_file, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info(f"记录: {line.strip()}")


def run_syng_bts_experiment(
    batch_dir,
    output_dir,
    dataname,
    model_name,
    off_aug,
    data_type,
    case_id,
    normalization_method,
    batch_num,
    record_file,
    total_experiments,
    completed_experiments,
):
    from datetime import datetime

    start_time = datetime.now()
    logger.info(
        f"开始: {case_id} | off_aug={off_aug} model={model_name} data_type={data_type}"
    )

    try:
        ApplyExperiment(
            path=None,
            dataname=dataname,
            apply_log=True,
            new_size=[1000],
            model=model_name,
            batch_frac=0.1,
            learning_rate=0.00015,
            epoch=10000,
            early_stop_num=200,
            off_aug=off_aug,
            AE_head_num=2,
            Gaussian_head_num=9,
            pre_model=None,
            save_model=None,
            data_dir=str(Path(batch_dir)),
            output_dir=str(Path(output_dir)),
            data_type=data_type,
        )
        end_time = datetime.now()
        write_experiment_record(
            record_file,
            case_id,
            normalization_method,
            batch_num,
            off_aug,
            model_name,
            data_type,
            "success",
            start_time,
            end_time,
            total_experiments,
            completed_experiments,
        )
        logger.info(f"完成: {case_id} {off_aug} {model_name} / {data_type}")
    except Exception as e:
        end_time = datetime.now()
        write_experiment_record(
            record_file,
            case_id,
            normalization_method,
            batch_num,
            off_aug,
            model_name,
            data_type,
            "fail",
            start_time,
            end_time,
            total_experiments,
            completed_experiments,
        )
        logger.error(f"失败: {case_id} {off_aug} {model_name} / {data_type}, 错误: {e}")
        raise


def process_single_dataset(
    data_path,
    case_id,
    normalization_method,
    base_output_path,
    record_file,
    off_augs,
    models,
    data_types,
    total_experiments,
    completed_experiments,
    n_batches,
):
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"读取失败: {data_path}, {e}")
        return completed_experiments

    if "samples" not in data.columns:
        logger.error(f"缺少列 samples: {data_path}")
        return completed_experiments
    if "groups" not in data.columns:
        logger.error(f"缺少列 groups: {data_path}")
        return completed_experiments

    logger.info(f"{case_id}_{normalization_method} 样本数: {len(data)}")
    for g, c in data["groups"].value_counts().items():
        logger.info(f"  {g}: {c}")

    for batch_num in range(1, n_batches + 1):
        logger.info(f"--- {case_id} {normalization_method} batch {batch_num}/{n_batches} ---")

        train_data, test_data = split_data_by_sample_size(
            data, case_id, normalization_method, batch_num
        )
        batch_dir = Path(base_output_path) / case_id / normalization_method / f"batch_{batch_num}"
        base_filename = f"{case_id}_{normalization_method}_batch_{batch_num}"
        save_split_data(train_data, test_data, batch_dir, base_filename)
        dataname = f"{base_filename}_train"

        for off_aug in off_augs:
            for model_name in models:
                for data_type in data_types:
                    output_dir = create_run_output_dir(
                        base_output_path,
                        case_id,
                        normalization_method,
                        batch_num,
                        off_aug,
                        model_name,
                        data_type,
                    )
                    run_syng_bts_experiment(
                        batch_dir,
                        output_dir,
                        dataname,
                        model_name,
                        off_aug,
                        data_type,
                        case_id,
                        normalization_method,
                        batch_num,
                        record_file,
                        total_experiments,
                        completed_experiments,
                    )
                    completed_experiments += 1
                    logger.info(
                        f"进度: {completed_experiments}/{total_experiments} "
                        f"({100 * completed_experiments / total_experiments:.1f}%)"
                    )

    return completed_experiments


def main():
    from datetime import datetime

    logger.info("PAAD-READ-SKCM 增强工作流开始 (SyNG-BTS 2.6)")

    data_case_root = _REPO_ROOT / "syng_bts" / "data" / "case"
    output_base_path = (
        _REPO_ROOT / "syng_bts" / "data" / "case" / "Augmentation-COAD-LAML-2026-03-30"
    )
    output_base_path.mkdir(parents=True, exist_ok=True)

    record_file = _LOG_DIR / "Complete_workflow_PAAD-READ-SKCM_records.txt"
    with open(record_file, "w", encoding="utf-8") as f:
        f.write("PAAD-READ-SKCM 增强实验记录 (SyNG-BTS 2.6)\n")
        f.write("格式: case_id-标准化-batch-off_aug-模型-data_type-状态|时间戳|耗时|完成比例|已完成/总数\n")
        f.write("=" * 120 + "\n")

    total_experiments = (
        len(CANCER_SUBTYPES)
        * len(NORMALIZATION_METHODS)
        * N_BATCHES
        * len(OFF_AUGS)
        * len(MODELS)
        * len(DATA_TYPES)
    )
    logger.info(f"总实验数: {total_experiments} (3×3×20×2×3×1)")
    logger.info(
        f"癌症: {[c[0] for c in CANCER_SUBTYPES]}, 标准化: {NORMALIZATION_METHODS}, "
        f"off_aug: {OFF_AUGS}, 模型: {MODELS}, data_type: {DATA_TYPES}, batches: {N_BATCHES}"
    )

    completed_experiments = 0
    for case_id, prefix in CANCER_SUBTYPES:
        data_base_path = data_case_root / case_id
        if not data_base_path.is_dir():
            logger.warning(f"跳过（目录不存在）: {data_base_path}")
            continue

        for normalization_method in NORMALIZATION_METHODS:
            data_filename = get_data_filename(prefix, normalization_method)
            data_path = data_base_path / data_filename
            if not data_path.exists():
                logger.warning(f"跳过（文件不存在）: {data_path}")
                continue

            logger.info("=" * 80)
            logger.info(f"处理 {case_id}_{normalization_method}")
            logger.info("=" * 80)
            completed_experiments = process_single_dataset(
                data_path,
                case_id,
                normalization_method,
                output_base_path,
                record_file,
                OFF_AUGS,
                MODELS,
                DATA_TYPES,
                total_experiments,
                completed_experiments,
                N_BATCHES,
            )

    with open(record_file, "a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"已完成: {completed_experiments}\n")
        f.write(f"完成率: {100 * completed_experiments / total_experiments:.1f}%\n")

    logger.info("=" * 80)
    logger.info("PAAD-READ-SKCM 增强工作流结束")
    logger.info(f"输出目录: {output_base_path}")
    logger.info(f"记录文件: {record_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
