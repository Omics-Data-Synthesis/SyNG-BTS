#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特定配置workflow脚本
"""

import os
import sys
import pandas as pd
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_files():
    """测试数据文件是否存在"""
    data_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/data"
    
    data_files = {
        "KICH": "KICHSubtype_lymph_nodes_examined_TC.csv",
        "READ": "READPositive_4_TC.csv", 
        "UCS": "UCSSubtype_primary_pathology_total_pelv_lnr_TC.csv"
    }
    
    logger.info("检查数据文件是否存在...")
    
    for cancer_type, data_file in data_files.items():
        data_path = os.path.join(data_base_path, cancer_type, data_file)
        if os.path.exists(data_path):
            logger.info(f"OK {cancer_type}: {data_path} exists")
            # 检查数据大小
            try:
                data = pd.read_csv(data_path)
                logger.info(f"  - 样本数: {len(data)}")
                logger.info(f"  - 特征数: {len(data.columns)}")
            except Exception as e:
                logger.error(f"  - 读取数据失败: {str(e)}")
        else:
            logger.error(f"ERROR {cancer_type}: {data_path} not found")

def test_output_directories():
    """测试输出目录创建"""
    output_base_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case"
    
    logger.info("检查输出目录...")
    
    for cancer_type in ["READ", "UCS", "KICH"]:
        test_path = os.path.join(output_base_path, cancer_type, "TC", "batch_1")
        try:
            os.makedirs(test_path, exist_ok=True)
            logger.info(f"OK Successfully created directory: {test_path}")
        except Exception as e:
            logger.error(f"ERROR Failed to create directory: {test_path}, error: {str(e)}")

def main():
    """主测试函数"""
    logger.info("开始测试特定配置workflow...")
    
    # 测试数据文件
    test_data_files()
    
    # 测试输出目录
    test_output_directories()
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main()