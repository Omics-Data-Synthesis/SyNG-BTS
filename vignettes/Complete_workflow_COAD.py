#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COAD 亚类增强完整工作流 (SyNG-BTS 2.6)
- 1 种癌症亚类: COAD
- 3 种标准化: raw / TC / DESeq
- 6 个 CVAE 模型: CVAE10-1, CVAE1-1, CVAE1-10, CVAE1-50, CVAE1-100, CVAE1-200
- 2 种 data_type: miRNA, RNA
- 10 个 batch
总实验数: 3 × 10 × 6 × 2 = 360 次
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# 确保使用本仓库 (SyNG-BTS_2.6) 的 syng_bts，而非其他 repo 的安装包
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 非交互式 matplotlib 后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", "")

from syng_bts.experiments import ApplyExperiment

# 日志
_LOG_DIR = _REPO_ROOT / "vignettes"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "Complete_workflow_COAD.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def split_data_by_sample_size(data, normalization_method, batch_num):
    """按样本量划分 train/test，并 shuffle 训练集。"""
    n_samples = len(data)
    if n_samples < 100:
        logger.info(f"COAD_5-2_{normalization_method}_batch_{batch_num}: n={n_samples}<100，使用全量数据")
        train_data = data.copy()
        test_data = data.copy()
    elif 100 <= n_samples < 200:
        logger.info(f"COAD_5-2_{normalization_method}_batch_{batch_num}: n={n_samples}, 50%-50% 划分")
        train_data, test_data = train_test_split(
            data,
            test_size=0.5,
            random_state=batch_num,
            stratify=data["groups"] if "groups" in data.columns else None,
        )
    else:
        logger.info(f"COAD_5-2_{normalization_method}_batch_{batch_num}: n={n_samples}, 80%-20% 划分")
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=batch_num,
            stratify=data["groups"] if "groups" in data.columns else None,
        )
    train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
    return train_data, test_data


def save_split_data(train_data, test_data, batch_dir, base_filename):
    """将 train/test 保存到 batch 目录（该 batch 下所有模型共用）。"""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    train_file = batch_dir / f"{base_filename}_train.csv"
    test_file = batch_dir / f"{base_filename}_test.csv"
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    logger.info(f"已保存: {train_file}, {test_file}")


def create_run_output_dir(base_path, normalization_method, batch_num, model_name, data_type):
    """创建单次实验输出目录: .../norm/batch_k/model_name/data_type"""
    output_path = Path(base_path) / normalization_method / f"batch_{batch_num}" / model_name / data_type
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_experiment_record(
    record_file,
    normalization_method,
    batch_num,
    model_name,
    data_type,
    status,
    start_time,
    end_time,
    total_experiments,
    completed_experiments,
):
    """写入单条实验记录。"""
    from datetime import datetime

    completion_ratio = (completed_experiments / total_experiments) * 100
    duration = end_time - start_time
    duration_str = str(duration).split(".")[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"COAD_5-2-{normalization_method}-batch{batch_num}-{model_name}-{data_type}-{status}"
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
    data_type,
    normalization_method,
    batch_num,
    record_file,
    total_experiments,
    completed_experiments,
):
    """调用 SyNG-BTS 2.6 的 ApplyExperiment 跑单次实验。"""
    from datetime import datetime

    start_time = datetime.now()
    logger.info(f"开始: {dataname} | model={model_name} data_type={data_type}")

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
            off_aug=None,
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
            normalization_method,
            batch_num,
            model_name,
            data_type,
            "success",
            start_time,
            end_time,
            total_experiments,
            completed_experiments,
        )
        logger.info(f"完成: {model_name} / {data_type}")
    except Exception as e:
        end_time = datetime.now()
        write_experiment_record(
            record_file,
            normalization_method,
            batch_num,
            model_name,
            data_type,
            "fail",
            start_time,
            end_time,
            total_experiments,
            completed_experiments,
        )
        logger.error(f"失败: {model_name} / {data_type}, 错误: {e}")
        raise


def process_single_dataset(
    data_path,
    normalization_method,
    base_output_path,
    record_file,
    models,
    data_types,
    total_experiments,
    completed_experiments,
    n_batches,
):
    """处理一种标准化下的所有 batch 与 (model × data_type) 组合。"""
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

    logger.info(f"COAD_5-2_{normalization_method} 样本数: {len(data)}")
    for g, c in data["groups"].value_counts().items():
        logger.info(f"  {g}: {c}")

    for batch_num in range(1, n_batches + 1):
        logger.info(f"--- batch {batch_num}/{n_batches} ---")

        train_data, test_data = split_data_by_sample_size(data, normalization_method, batch_num)
        batch_dir = Path(base_output_path) / normalization_method / f"batch_{batch_num}"
        base_filename = f"COAD_5-2_{normalization_method}_batch_{batch_num}"
        save_split_data(train_data, test_data, batch_dir, base_filename)
        dataname = f"{base_filename}_train"

        for model_name in models:
            for data_type in data_types:
                output_dir = create_run_output_dir(
                    base_output_path, normalization_method, batch_num, model_name, data_type
                )
                run_syng_bts_experiment(
                    batch_dir,
                    output_dir,
                    dataname,
                    model_name,
                    data_type,
                    normalization_method,
                    batch_num,
                    record_file,
                    total_experiments,
                    completed_experiments,
                )
                completed_experiments += 1
                logger.info(f"进度: {completed_experiments}/{total_experiments} ({100 * completed_experiments / total_experiments:.1f}%)")

    return completed_experiments


def main():
    from datetime import datetime

    logger.info("COAD 增强工作流开始 (SyNG-BTS 2.6)")

    data_base_path = _REPO_ROOT / "syng_bts" / "data" / "case" / "COAD_5-2"
    output_base_path = _REPO_ROOT / "syng_bts" / "data" / "case" / "Augmentation-COAD-2026-02-23"
    output_base_path.mkdir(parents=True, exist_ok=True)

    record_file = _LOG_DIR / "Complete_workflow_COAD_records.txt"
    with open(record_file, "w", encoding="utf-8") as f:
        f.write("COAD 增强实验记录 (SyNG-BTS 2.6)\n")
        f.write("格式: COAD_5-2-标准化-batch-模型-data_type-状态|时间戳|耗时|完成比例|已完成/总数\n")
        f.write("=" * 120 + "\n")

    normalization_methods = ["raw", "TC", "DESeq"]
    data_files = {
        "raw": "COADPositive_5-2.csv",
        "TC": "COADPositive_5-2_TC.csv",
        "DESeq": "COADPositive_5-2_DESeq.csv",
    }
    models = ["CVAE10-1", "CVAE1-1", "CVAE1-10", "CVAE1-50", "CVAE1-100", "CVAE1-200"]
    data_types = ["miRNA", "RNA"]
    n_batches = 10

    total_experiments = len(normalization_methods) * n_batches * len(models) * len(data_types)
    logger.info(f"总实验数: {total_experiments} (3×10×6×2)")
    logger.info(f"标准化: {normalization_methods}, 模型: {models}, data_type: {data_types}, batches: {n_batches}")

    completed_experiments = 0
    for normalization_method in normalization_methods:
        data_path = data_base_path / data_files[normalization_method]
        if not data_path.exists():
            logger.warning(f"跳过（文件不存在）: {data_path}")
            continue
        logger.info("=" * 80)
        logger.info(f"处理 COAD_5-2_{normalization_method}")
        logger.info("=" * 80)
        completed_experiments = process_single_dataset(
            data_path,
            normalization_method,
            output_base_path,
            record_file,
            models,
            data_types,
            total_experiments,
            completed_experiments,
            n_batches,
        )

    with open(record_file, "a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"已完成: {completed_experiments}\n")
        f.write(f"完成率: {100 * completed_experiments / total_experiments:.1f}%\n")

    logger.info("=" * 80)
    logger.info("COAD 增强工作流结束")
    logger.info(f"记录文件: {record_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
