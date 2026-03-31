# -*- coding: utf-8 -*-
import sys
import os

# 设置matplotlib后端为非交互式，防止显示plot窗口
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式

# 设置环境变量，确保不会显示图形界面
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

from syng_bts.python.Experiments_new import *

# # 确保使用本地版本的模块
# # 方法1: 将当前项目根目录添加到Python路径的最前面
# project_root = "/Users/yanjiechen/Documents/Github/SyNG-BTS"
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # 方法2: 直接导入本地文件
# import importlib.util
# spec = importlib.util.spec_from_file_location(
#     "Experiments_new", 
#     "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/python/Experiments_new.py"
# )
# Experiments_new = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(Experiments_new)

# # 从本地模块导入ApplyExperiment
# ApplyExperiment = Experiments_new.ApplyExperiment

# # 添加调试信息，确认调用的是本地版本
# print(f"ApplyExperiment 函数来源: {ApplyExperiment.__code__.co_filename}")
# print(f"ApplyExperiment 函数参数: {ApplyExperiment.__code__.co_varnames}")

# ApplyExperiment(path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case/KICH/raw/batch_1/",
#                 dataname = "KICH_raw_batch_1_train", 
#                 apply_log = True, 
#                 new_size = [1000],
#                 model = "CVAE1-10" ,
#                 batch_frac = 0.1, 
#                 learning_rate = 0.0005,
#                 epoch = 3000,
#                 early_stop_num = 20, 
#                 off_aug = "AE_head",
#                 AE_head_num = 2,
#                 Gaussian_head_num = 9, 
#                 pre_model = None,
#                 save_model = None,
#                 random_seed = 123)


ApplyExperiment(path = "/Users/yanjiechen/Documents/Github/SyNG-BTS/syng_bts/Case/KICH/raw/batch_2/",
                dataname = "KICH_raw_batch_2_train", 
                apply_log = True, 
                new_size = [1000],
                model = "CVAE1-10" ,
                batch_frac = 0.1, 
                learning_rate = 0.0005,
                epoch = 3000,
                early_stop_num = 20, 
                off_aug = "AE_head",
                AE_head_num = 2,
                Gaussian_head_num = 9, 
                pre_model = None,
                save_model = None)