#!/bin/bash
# 运行特定配置的SyNG-BTS workflow脚本

echo "开始运行特定配置的SyNG-BTS workflow..."
echo "配置: READ/UCS/KICH + TC标准化 + CVAE1-10模型 + 三种off_aug方法"
echo "总实验数: 3种癌症类型 × 20个batch × 3种增强方法 = 180次实验"
echo ""

# 切换到项目根目录
cd /Users/yanjiechen/Documents/Github/SyNG-BTS

# 运行脚本
python3 syng_bts/python/specific_config_workflow.py

echo ""
echo "特定配置workflow完成！"
echo "请查看实验记录文件: syng_bts/python/specific_config_experiment_records.txt"
