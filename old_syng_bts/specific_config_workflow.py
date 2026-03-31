#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特定配置的SyNG-BTS数据增强workflow脚本
处理指定的癌症类型和配置：
1. READ: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]
2. UCS: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]  
3. KICH: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

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
        logging.FileHandler('/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/python/specific_config_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data_for_paad_read(data, cancer_type):
    """
    为PAAD和READ癌症类型预处理数据：裁剪samples列并添加groups列
    
    Parameters:
    -----------
    data : pandas.DataFrame
        输入数据
    cancer_type : str
        癌症类型 ("PAAD" 或 "READ")
        
    Returns:
    --------
    processed_data : pandas.DataFrame
        处理后的数据
    """
    if cancer_type not in ["PAAD", "READ"]:
        logger.info(f"{cancer_type} 不需要特殊预处理，直接返回原数据")
        return data
    
    logger.info(f"开始预处理 {cancer_type} 数据")
    
    # 1. 裁剪samples列，仅保留前12个字符
    if 'samples' in data.columns:
        data['samples'] = data['samples'].astype(str).str[:12]
        logger.info(f"已裁剪samples列，保留前12个字符")
    else:
        logger.warning(f"数据中未找到samples列")
        return data
    
    # 2. 读取对应的患者信息文件
    if cancer_type == "PAAD":
        patient_file = "/Users/yanjiechen/Documents/Github/SyNG-BTS/data/PAAD/PAADPatientFilter.csv"
        group_column = "anatomic_neoplasm_subdivision"
    elif cancer_type == "READ":
        patient_file = "/Users/yanjiechen/Documents/Github/SyNG-BTS/data/READ/READPatientFilter.csv"
        group_column = "ajcc_staging_edition"
    
    try:
        patient_data = pd.read_csv(patient_file)
        logger.info(f"成功读取患者信息文件: {patient_file}")
    except Exception as e:
        logger.error(f"读取患者信息文件失败: {patient_file}, 错误: {str(e)}")
        return data
    
    # 3. 创建samples到groups的映射
    # 将samples中的"."替换为"-"以匹配patient文件中的格式
    data['samples_normalized'] = data['samples'].str.replace('.', '-')
    
    # 创建映射字典
    sample_to_group = {}
    for _, row in patient_data.iterrows():
        bcr_code = row['bcr_patient_barcode']
        group_value = row[group_column]
        sample_to_group[bcr_code] = group_value
    
    # 4. 为数据添加groups列
    data['groups'] = data['samples_normalized'].map(sample_to_group)
    
    # 处理NaN值：直接删除无法匹配的样本
    original_count = len(data)
    data = data.dropna(subset=['groups'])
    removed_count = original_count - len(data)
    
    if removed_count > 0:
        logger.warning(f"删除了 {removed_count} 个无法匹配到患者信息的样本")
        logger.info(f"剩余样本数: {len(data)}")
    
    # 统计分组情况
    group_counts = data['groups'].value_counts()
    logger.info(f"{cancer_type} 数据分组统计:")
    for group, count in group_counts.items():
        logger.info(f"  {group}: {count} 个样本")
    
    # 删除临时的samples_normalized列
    data = data.drop('samples_normalized', axis=1)
    
    logger.info(f"{cancer_type} 数据预处理完成，总样本数: {len(data)}")
    return data

def split_data_by_sample_size(data, cancer_type, batch_num):
    """
    根据样本数量决定数据分割策略
    
    Parameters:
    -----------
    data : pandas.DataFrame
        输入数据
    cancer_type : str
        癌症类型
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
        logger.info(f"{cancer_type}_batch_{batch_num}: 样本数{n_samples}<100，使用完整数据")
        train_data = data.copy()
        test_data = data.copy()
        
        # 对训练数据进行shuffle
        train_data = train_data.sample(frac=1, random_state=batch_num).reset_index(drop=True)
        logger.info(f"已对训练数据进行shuffle")
        
        return train_data, test_data
    elif 100 <= n_samples < 200:
        # 样本数100-200，按50%-50%分割
        logger.info(f"{cancer_type}_batch_{batch_num}: 样本数{n_samples}在100-200之间，按50%-50%分割")
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
        logger.info(f"{cancer_type}_batch_{batch_num}: 样本数{n_samples}>=200，按80%-20%分割")
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

def create_output_directories(base_path, cancer_type, batch_num):
    """
    创建输出目录结构
    
    Parameters:
    -----------
    base_path : str
        基础路径
    cancer_type : str
        癌症类型
    batch_num : int
        batch编号
        
    Returns:
    --------
    output_path : str
        输出路径
    """
    output_path = os.path.join(base_path, cancer_type, "TC", f"batch_{batch_num}")
    os.makedirs(output_path, exist_ok=True)
    return output_path

def save_split_data(train_data, test_data, output_path, cancer_type, batch_num):
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
    batch_num : int
        batch编号
    """
    # 生成文件名
    base_filename = f"{cancer_type}_TC_batch_{batch_num}"
    
    # 保存训练数据
    train_file = os.path.join(output_path, f"{base_filename}_train.csv")
    train_data.to_csv(train_file, index=False)
    logger.info(f"保存训练数据: {train_file}")
    
    # 保存测试数据
    test_file = os.path.join(output_path, f"{base_filename}_test.csv")
    test_data.to_csv(test_file, index=False)
    logger.info(f"保存测试数据: {test_file}")

def write_experiment_record(record_file, cancer_type, batch_num, 
                          off_aug_method, status, start_time, end_time, total_experiments, completed_experiments):
    """
    写入实验记录到txt文件
    
    Parameters:
    -----------
    record_file : str
        记录文件路径
    cancer_type : str
        癌症类型
    batch_num : int
        batch编号
    off_aug_method : str
        增强方法
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
    # 计算完成比例
    completion_ratio = (completed_experiments / total_experiments) * 100
    
    # 计算实验耗时
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]  # 去掉微秒部分
    
    # 格式化时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建记录行
    record_line = f"{cancer_type}-TC-batch{batch_num}-{off_aug_method}-{status}|{timestamp}|{duration_str}|{completion_ratio:.1f}%|{completed_experiments}/{total_experiments}\n"
    
    # 写入文件
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(record_line)
    
    logger.info(f"实验记录已写入: {record_line.strip()}")

def run_syng_bts_experiment(output_path, cancer_type, batch_num, 
                          off_aug_method, record_file, total_experiments, completed_experiments):
    """
    运行SyNG-BTS实验
    
    Parameters:
    -----------
    output_path : str
        输出路径
    cancer_type : str
        癌症类型
    batch_num : int
        batch编号
    off_aug_method : str
        增强方法 ("AE_head", "Gaussian" 或 None)
    record_file : str
        记录文件路径
    total_experiments : int
        总实验数
    completed_experiments : int
        已完成实验数
    """
    base_filename = f"{cancer_type}_TC_batch_{batch_num}"
    dataname = f"{base_filename}_train"
    
    # 记录开始时间
    start_time = datetime.now()
    
    logger.info(f"开始实验: {cancer_type}_TC_batch_{batch_num}_{off_aug_method}")
    
    try:
        # 根据off_aug_method设置不同的参数
        if off_aug_method == "Gaussian":
            # 对于Gaussian方法，使用Gaussian_head
            ApplyExperiment(
                path=output_path + "/",
                dataname=dataname,
                apply_log=True,
                new_size=[1000],
                model="CVAE1-10",
                batch_frac=0.1,
                learning_rate=0.0005,
                epoch=3000,
                early_stop_num=20,
                off_aug="Gaussian_head",  # 使用Gaussian_head
                AE_head_num=2,
                Gaussian_head_num=9,
                pre_model=None,
                save_model=None
            )
        else:
            # 对于AE_head和none方法
            ApplyExperiment(
                path=output_path + "/",
                dataname=dataname,
                apply_log=True,
                new_size=[1000],
                model="CVAE1-10",
                batch_frac=0.1,
                learning_rate=0.0005,
                epoch=3000,
                early_stop_num=20,
                off_aug=off_aug_method,
                AE_head_num=2,
                Gaussian_head_num=9,
                pre_model=None,
                save_model=None
            )
        
        # 记录结束时间和成功状态
        end_time = datetime.now()
        write_experiment_record(record_file, cancer_type, batch_num, 
                              off_aug_method, "success", start_time, end_time, total_experiments, completed_experiments)
        
        logger.info(f"实验完成: {cancer_type}_TC_batch_{batch_num}_{off_aug_method}")
        
    except Exception as e:
        # 记录结束时间和失败状态
        end_time = datetime.now()
        write_experiment_record(record_file, cancer_type, batch_num, 
                              off_aug_method, "fail", start_time, end_time, total_experiments, completed_experiments)
        
        logger.error(f"实验失败: {cancer_type}_TC_batch_{batch_num}_{off_aug_method}, 错误: {str(e)}")

def process_single_dataset(data_path, cancer_type, base_output_path, 
                          record_file, total_experiments, completed_experiments):
    """
    处理单个数据集的所有batch
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    cancer_type : str
        癌症类型
    base_output_path : str
        基础输出路径
    record_file : str
        记录文件路径
    total_experiments : int
        总实验数
    completed_experiments : int
        已完成实验数
    """
    logger.info(f"开始处理数据集: {cancer_type}_TC")
    
    # 读取数据
    try:
        data = pd.read_csv(data_path)
        logger.info(f"成功读取数据: {data_path}, 样本数: {len(data)}")
    except Exception as e:
        logger.error(f"读取数据失败: {data_path}, 错误: {str(e)}")
        return completed_experiments
    
    # 对READ数据进行预处理
    data = preprocess_data_for_paad_read(data, cancer_type)
    
    # 处理20个batch
    for batch_num in range(1, 21):
        logger.info(f"处理batch {batch_num}/20")
        
        # 分割数据
        train_data, test_data = split_data_by_sample_size(
            data, cancer_type, batch_num
        )
        
        # 创建输出目录
        output_path = create_output_directories(
            base_output_path, cancer_type, batch_num
        )
        
        # 保存分割数据
        save_split_data(train_data, test_data, output_path, 
                       cancer_type, batch_num)
        
        # 运行三种增强方法的实验
        for off_aug_method in [None, "AE_head", "Gaussian"]:
            run_syng_bts_experiment(
                output_path, cancer_type, batch_num,
                off_aug_method, record_file, total_experiments, completed_experiments
            )
            
            # 更新已完成实验计数
            completed_experiments += 1
    
    return completed_experiments

def main():
    """
    主函数：运行特定配置的workflow
    """
    logger.info("开始特定配置的SyNG-BTS workflow")
    
    # 定义数据路径和参数
    data_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/data"
    output_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case"
    
    # 创建实验记录文件
    record_file = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/python/specific_config_experiment_records.txt"
    
    # 初始化记录文件
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write("特定配置SyNG-BTS实验记录\n")
        f.write("配置: READ/UCS/KICH + TC标准化 + CVAE1-10模型 + 三种off_aug方法\n")
        f.write("格式: 癌症种类-TC-batch编号-增强方法-状态|时间戳|耗时|完成比例|已完成/总数\n")
        f.write("=" * 100 + "\n")
    
    logger.info(f"实验记录文件已创建: {record_file}")
    
    # 定义特定的癌症类型和配置
    cancer_types = ["READ", "UCS", "KICH"]
    
    # 定义数据文件映射（只使用TC标准化方法）
    data_files = {
        "KICH": "KICHSubtype_lymph_nodes_examined_TC.csv",
        "READ": "READPositive_4_TC.csv", 
        "UCS": "UCSSubtype_primary_pathology_total_pelv_lnr_TC.csv"
    }
    
    total_experiments = len(cancer_types) * 20 * 3  # 3种癌症类型 × 20个batch × 3种增强方法
    logger.info(f"总共需要运行 {total_experiments} 次实验")
    
    completed_experiments = 0
    
    # 遍历所有癌症类型
    for cancer_type in cancer_types:
        data_file = data_files[cancer_type]
        data_path = os.path.join(data_base_path, cancer_type, data_file)
        
        if not os.path.exists(data_path):
            logger.warning(f"数据文件不存在: {data_path}")
            continue
            
        logger.info(f"处理: {cancer_type}_TC")
        
        # 处理单个数据集
        completed_experiments = process_single_dataset(
            data_path, cancer_type, output_base_path,
            record_file, total_experiments, completed_experiments
        )
        
        logger.info(f"已完成 {completed_experiments}/{total_experiments} 次实验")
    
    # 写入最终统计信息
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"已完成实验数: {completed_experiments}\n")
        f.write(f"完成率: {(completed_experiments/total_experiments)*100:.1f}%\n")
    
    logger.info("特定配置的SyNG-BTS workflow完成！")
    logger.info(f"实验记录已保存到: {record_file}")

if __name__ == "__main__":
    main()
