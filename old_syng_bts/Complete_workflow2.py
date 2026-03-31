#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新数据集测试脚本
处理5种癌症类型 × 3种标准化方法 × 20个batch × 3种CVAE模型 = 900次实验
数据已经标准化了samples列、加上了group variable、并去掉了NaN的行
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# 设置matplotlib后端为非交互式，防止显示plot窗口
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式

# 设置环境变量，确保不会显示图形界面
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

# 导入SyNG-BTS实验函数
from syng_bts.python.Experiments_new import ApplyExperiment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/python/test_new_datasets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def split_data_by_sample_size(data, cancer_type, normalization_method, batch_num):
    """
    根据样本数量决定数据分割策略
    
    Parameters:
    -----------
    data : pandas.DataFrame
        输入数据
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    batch_num : int
        batch编号
        
    Returns:
    --------
    train_data, test_data : pandas.DataFrame
        训练和测试数据
    """
    n_samples = len(data)
    
    if n_samples < 100:
        # 样本数<100，使用完整数据作为train和test，并对train data进行shuffle
        logger.info(f"{cancer_type}_{normalization_method}_batch_{batch_num}: 样本数{n_samples}<100，使用完整数据")
        train_data = data.copy()
        test_data = data.copy()
        
        # 对训练数据进行shuffle
        train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
        logger.info(f"已对训练数据进行shuffle")
        
        return train_data, test_data
    elif 100 <= n_samples < 200:
        # 样本数100-200，按50%-50%分割
        logger.info(f"{cancer_type}_{normalization_method}_batch_{batch_num}: 样本数{n_samples}在100-200之间，按50%-50%分割")
        train_data, test_data = train_test_split(
            data, 
            test_size=0.5, 
            random_state=batch_num,  # 使用batch_num作为随机种子
            stratify=data['groups'] if 'groups' in data.columns else None
        )
        
        # 对训练数据进行shuffle
        train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
        logger.info(f"已对训练数据进行shuffle")
        
        return train_data, test_data
    else:
        # 样本数>=200，按80%-20%分割
        logger.info(f"{cancer_type}_{normalization_method}_batch_{batch_num}: 样本数{n_samples}>=200，按80%-20%分割")
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=batch_num,  # 使用batch_num作为随机种子
            stratify=data['groups'] if 'groups' in data.columns else None
        )
        
        # 对训练数据进行shuffle
        train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
        logger.info(f"已对训练数据进行shuffle")
        
        return train_data, test_data

def create_output_directories(base_path, cancer_type, normalization_method, batch_num, model_name):
    """
    创建输出目录结构
    
    Parameters:
    -----------
    base_path : str
        基础路径
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    batch_num : int
        batch编号
    model_name : str
        模型名称
        
    Returns:
    --------
    output_path : str
        输出路径
    """
    output_path = os.path.join(base_path, f"{cancer_type}_5-2", normalization_method, f"batch_{batch_num}", model_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def save_split_data(train_data, test_data, output_path, cancer_type, normalization_method, batch_num):
    """
    保存分割后的数据
    
    Parameters:
    -----------
    train_data, test_data : pandas.DataFrame
        训练和测试数据
    output_path : str
        输出路径
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    batch_num : int
        batch编号
    """
    # 生成文件名
    base_filename = f"{cancer_type}_5-2_{normalization_method}_batch_{batch_num}"
    
    # 保存训练数据
    train_file = os.path.join(output_path, f"{base_filename}_train.csv")
    train_data.to_csv(train_file, index=False)
    logger.info(f"保存训练数据: {train_file}")
    
    # 保存测试数据
    test_file = os.path.join(output_path, f"{base_filename}_test.csv")
    test_data.to_csv(test_file, index=False)
    logger.info(f"保存测试数据: {test_file}")

def write_experiment_record(record_file, cancer_type, normalization_method, batch_num, 
                          model_name, status, start_time, end_time, total_experiments, completed_experiments):
    """
    写入实验记录到txt文件
    
    Parameters:
    -----------
    record_file : str
        记录文件路径
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    batch_num : int
        batch编号
    model_name : str
        模型名称
    status : str
        实验状态 (success/fail)
    start_time : datetime
        开始时间
    end_time : datetime
        结束时间
    total_experiments : int
        总实验数
    completed_experiments : int
        已完成实验数
    """
    from datetime import datetime
    
    # 计算完成比例
    completion_ratio = (completed_experiments / total_experiments) * 100
    
    # 计算实验耗时
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]  # 去掉微秒部分
    
    # 格式化时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建记录行
    record_line = f"{cancer_type}_5-2-{normalization_method}-batch{batch_num}-{model_name}-{status}|{timestamp}|{duration_str}|{completion_ratio:.1f}%|{completed_experiments}/{total_experiments}\n"
    
    # 写入文件
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(record_line)
    
    logger.info(f"实验记录已写入: {record_line.strip()}")

def run_syng_bts_experiment(output_path, cancer_type, normalization_method, batch_num, 
                          model_name, record_file, total_experiments, completed_experiments):
    """
    运行SyNG-BTS实验
    
    Parameters:
    -----------
    output_path : str
        输出路径
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    batch_num : int
        batch编号
    model_name : str
        模型名称 ("CVAE1-50", "CVAE1-100", "CVAE1-150")
    record_file : str
        记录文件路径
    total_experiments : int
        总实验数
    completed_experiments : int
        已完成实验数
    """
    from datetime import datetime
    
    base_filename = f"{cancer_type}_5-2_{normalization_method}_batch_{batch_num}"
    dataname = f"{base_filename}_train"
    
    # 记录开始时间
    start_time = datetime.now()
    
    logger.info(f"开始实验: {cancer_type}_5-2_{normalization_method}_batch_{batch_num}_{model_name}")
    
    try:
        ApplyExperiment(
            path=output_path + "/",
            dataname=dataname,
            apply_log=True,
            new_size=[1000],
            model=model_name,
            batch_frac=0.1,
            learning_rate=0.0005,
            epoch=3000,
            early_stop_num=20,
            off_aug=None,  # 设置为None，不使用AE head
            AE_head_num=2,
            Gaussian_head_num=9,
            pre_model=None,
            save_model=None
        )
        
        # 记录结束时间和成功状态
        end_time = datetime.now()
        write_experiment_record(record_file, cancer_type, normalization_method, batch_num, 
                              model_name, "success", start_time, end_time, total_experiments, completed_experiments)
        
        logger.info(f"实验完成: {cancer_type}_5-2_{normalization_method}_batch_{batch_num}_{model_name}")
        
    except Exception as e:
        # 记录结束时间和失败状态
        end_time = datetime.now()
        write_experiment_record(record_file, cancer_type, normalization_method, batch_num, 
                              model_name, "fail", start_time, end_time, total_experiments, completed_experiments)
        
        logger.error(f"实验失败: {cancer_type}_5-2_{normalization_method}_batch_{batch_num}_{model_name}, 错误: {str(e)}")
        raise e  # 重新抛出异常以便调试

def process_single_dataset(data_path, cancer_type, normalization_method, base_output_path, 
                          record_file, models, total_experiments, completed_experiments):
    """
    处理单个数据集的所有batch和模型
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    cancer_type : str
        癌症类型
    normalization_method : str
        标准化方法
    base_output_path : str
        基础输出路径
    record_file : str
        记录文件路径
    models : list
        模型列表
    total_experiments : int
        总实验数
    completed_experiments : int
        已完成实验数
    """
    logger.info(f"开始处理数据集: {cancer_type}_5-2_{normalization_method}")
    
    # 读取数据
    try:
        data = pd.read_csv(data_path)
        logger.info(f"成功读取数据: {data_path}, 样本数: {len(data)}")
        
        # 检查数据是否包含必要的列
        if 'samples' not in data.columns:
            logger.error(f"数据缺少samples列: {data_path}")
            return completed_experiments
        
        if 'groups' not in data.columns:
            logger.error(f"数据缺少groups列: {data_path}")
            return completed_experiments
            
        # 统计分组情况
        group_counts = data['groups'].value_counts()
        logger.info(f"{cancer_type}_5-2 数据分组统计:")
        for group, count in group_counts.items():
            logger.info(f"  {group}: {count} 个样本")
            
    except Exception as e:
        logger.error(f"读取数据失败: {data_path}, 错误: {str(e)}")
        return completed_experiments
    
    # 处理20个batch
    for batch_num in range(1, 21):
        logger.info(f"处理batch {batch_num}/20")
        
        # 分割数据
        train_data, test_data = split_data_by_sample_size(
            data, cancer_type, normalization_method, batch_num
        )
        
        # 运行三种模型的实验
        for model_name in models:
            # 创建输出目录（每个模型一个子目录）
            output_path = create_output_directories(
                base_output_path, cancer_type, normalization_method, batch_num, model_name
            )
            
            # 保存分割数据
            save_split_data(train_data, test_data, output_path, 
                           cancer_type, normalization_method, batch_num)
            
            # 运行实验
            run_syng_bts_experiment(
                output_path, cancer_type, normalization_method, batch_num,
                model_name, record_file, total_experiments, completed_experiments
            )
            
            # 更新已完成实验计数
            completed_experiments += 1
            logger.info(f"进度: {completed_experiments}/{total_experiments} ({(completed_experiments/total_experiments)*100:.1f}%)")
    
    return completed_experiments

def main():
    """
    主函数：运行完整的测试workflow
    """
    from datetime import datetime
    
    logger.info("开始新数据集测试workflow")
    
    # 定义数据路径和参数
    data_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case"
    output_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case/test_results"
    
    # 创建输出基础目录
    os.makedirs(output_base_path, exist_ok=True)
    
    # 创建实验记录文件
    record_file = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/python/test_new_datasets_records.txt"
    
    # 初始化记录文件
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write("新数据集测试实验记录\n")
        f.write("格式: 癌症种类-标准化方法-batch编号-模型-状态|时间戳|耗时|完成比例|已完成/总数\n")
        f.write("=" * 120 + "\n")
    
    logger.info(f"实验记录文件已创建: {record_file}")
    
    # 定义癌症类型、标准化方法和模型
    cancer_types = ["COAD", "LAML", "PAAD", "READ", "SKCM"]
    normalization_methods = ["raw", "TC", "DESeq"]
    models = ["CVAE1-50", "CVAE1-100", "CVAE1-150"]
    
    # 定义数据文件映射
    data_files = {
        "COAD": {
            "raw": "COADPositive_5-2.csv",
            "TC": "COADPositive_5-2_TC.csv",
            "DESeq": "COADPositive_5-2_DESeq.csv"
        },
        "LAML": {
            "raw": "LAMLPositive_5-2.csv",
            "TC": "LAMLPositive_5-2_TC.csv",
            "DESeq": "LAMLPositive_5-2_DESeq.csv"
        },
        "PAAD": {
            "raw": "PAADPositive_5-2.csv",
            "TC": "PAADPositive_5-2_TC.csv",
            "DESeq": "PAADPositive_5-2_DESeq.csv"
        },
        "READ": {
            "raw": "READPositive_5-2.csv",
            "TC": "READPositive_5-2_TC.csv",
            "DESeq": "READPositive_5-2_DESeq.csv"
        },
        "SKCM": {
            "raw": "SKCMPositive_5-2.csv",
            "TC": "SKCMPositive_5-2_TC.csv",
            "DESeq": "SKCMPositive_5-2_DESeq.csv"
        }
    }
    
    # 计算总实验数: 5种癌症 × 3种标准化 × 20个batch × 3种模型 = 900次
    total_experiments = len(cancer_types) * len(normalization_methods) * 20 * len(models)
    logger.info(f"总共需要运行 {total_experiments} 次实验")
    logger.info(f"癌症类型: {cancer_types}")
    logger.info(f"标准化方法: {normalization_methods}")
    logger.info(f"模型: {models}")
    logger.info(f"每个数据集的batch数: 20")
    
    completed_experiments = 0
    
    # 遍历所有癌症类型和标准化方法
    for cancer_type in cancer_types:
        for normalization_method in normalization_methods:
            data_file = data_files[cancer_type][normalization_method]
            data_path = os.path.join(data_base_path, f"{cancer_type}_5-2", data_file)
            
            if not os.path.exists(data_path):
                logger.warning(f"数据文件不存在: {data_path}")
                # 跳过不存在的文件，但不增加completed_experiments
                # 因为这些实验不应该被计入总数
                continue
                
            logger.info(f"=" * 100)
            logger.info(f"处理: {cancer_type}_5-2_{normalization_method}")
            logger.info(f"=" * 100)
            
            # 处理单个数据集
            completed_experiments = process_single_dataset(
                data_path, cancer_type, normalization_method, output_base_path,
                record_file, models, total_experiments, completed_experiments
            )
            
            logger.info(f"已完成 {completed_experiments}/{total_experiments} 次实验")
    
    # 写入最终统计信息
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"已完成实验数: {completed_experiments}\n")
        f.write(f"完成率: {(completed_experiments/total_experiments)*100:.1f}%\n")
    
    logger.info("=" * 100)
    logger.info("新数据集测试workflow完成！")
    logger.info(f"实验记录已保存到: {record_file}")
    logger.info("=" * 100)

if __name__ == "__main__":
    main()
