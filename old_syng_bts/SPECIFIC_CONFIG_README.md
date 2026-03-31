# 特定配置SyNG-BTS数据增强Workflow

## 概述

这个脚本专门为特定的配置运行SyNG-BTS数据增强实验，而不是运行完整的workflow。

## 配置详情

脚本将运行以下特定配置：

1. **READ**: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]
2. **UCS**: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]  
3. **KICH**: model = CVAE1-10, norm = TC, off_aug = [none, AE_head, Gaussian]

## 实验规模

- **癌症类型**: 3种 (READ, UCS, KICH)
- **标准化方法**: 1种 (TC)
- **Batch数量**: 20个
- **增强方法**: 3种 (none, AE_head, Gaussian)
- **总实验数**: 3 × 20 × 3 = 180次实验

## 文件结构

```
syng_bts/python/
├── specific_config_workflow.py          # 主脚本
├── test_specific_config.py             # 测试脚本
├── run_specific_config.sh              # 运行脚本
├── specific_config_workflow.log         # 运行日志
└── specific_config_experiment_records.txt  # 实验记录
```

## 使用方法

### 方法1: 直接运行Python脚本

```bash
cd /Users/yanjiechen/Documents/Github/SyNG-BTS
python3 syng_bts/python/specific_config_workflow.py
```

### 方法2: 使用运行脚本

```bash
cd /Users/yanjiechen/Documents/Github/SyNG-BTS
./syng_bts/python/run_specific_config.sh
```

### 方法3: 测试功能

```bash
cd /Users/yanjiechen/Documents/Github/SyNG-BTS
python3 syng_bts/python/test_specific_config.py
```

## 输出结构

实验结果将保存在以下目录结构中：

```
syng_bts/Case/
├── READ/TC/batch_1/
│   ├── READ_TC_batch_1_train.csv
│   ├── READ_TC_batch_1_test.csv
│   └── [生成的增强数据文件]
├── READ/TC/batch_2/
│   └── ...
├── UCS/TC/batch_1/
│   └── ...
└── KICH/TC/batch_1/
    └── ...
```

## 实验记录格式

实验记录保存在 `specific_config_experiment_records.txt` 文件中，格式如下：

```
癌症种类-TC-batch编号-增强方法-状态|时间戳|耗时|完成比例|已完成/总数
```

示例：
```
READ-TC-batch1-AE_head-success|2024-01-01 12:00:00|00:05:30|5.6%|10/180
```

## 数据预处理

- **READ数据**: 会自动进行预处理，包括samples列裁剪和添加groups列
- **UCS和KICH数据**: 直接使用原始数据，无需特殊预处理

## 数据分割策略

根据样本数量自动选择分割策略：

- **< 100样本**: 使用完整数据作为训练和测试数据
- **100-200样本**: 按50%-50%分割
- **≥ 200样本**: 按80%-20%分割

## 增强方法说明

1. **none**: 不使用离线增强
2. **AE_head**: 使用AutoEncoder头部增强
3. **Gaussian**: 使用高斯增强方法

## 监控和日志

- **控制台输出**: 实时显示实验进度
- **日志文件**: `specific_config_workflow.log` 记录详细日志
- **实验记录**: `specific_config_experiment_records.txt` 记录实验结果

## 注意事项

1. 确保使用 `python3` 而不是 `python` 运行脚本
2. 实验可能需要较长时间完成（取决于数据大小和计算资源）
3. 建议在运行前先执行测试脚本验证环境
4. 实验过程中可以随时查看日志文件了解进度

## 故障排除

如果遇到问题，请检查：

1. 数据文件是否存在
2. 输出目录是否有写入权限
3. Python环境和依赖包是否正确安装
4. 查看日志文件中的错误信息
