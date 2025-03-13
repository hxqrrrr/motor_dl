import os
import sys
import subprocess
import json
import time
from datetime import datetime
import argparse
from utils.gpu_info import set_gpu, print_gpu_info

def run_experiment(command, gpu_id=None, log_dir="logs"):
    """
    运行单个实验命令
    
    参数:
        command: 要运行的命令
        gpu_id: 指定的GPU ID，如果为None则自动选择最佳GPU
        log_dir: 日志保存目录
    
    返回:
        bool: 实验是否成功
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置GPU
    if gpu_id is not None:
        # 在命令中添加GPU选择
        if "--gpu" in command:
            # 如果命令中已经有--gpu参数，替换它
            command = command.replace("--gpu", f"--gpu {gpu_id} --")
        else:
            # 否则添加--gpu参数
            command += f" --gpu {gpu_id}"
    
    # 获取当前时间作为日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    print(f"运行命令: {command}")
    print(f"日志将保存到: {log_file}")
    
    # 添加内存优化环境变量
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 打开日志文件
    with open(log_file, "w") as f:
        # 记录实验开始时间和命令
        f.write(f"实验开始时间: {timestamp}\n")
        f.write(f"命令: {command}\n\n")
        f.flush()
        
        # 运行命令并实时记录输出
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env  # 使用修改后的环境变量
        )
        
        # 实时读取并记录输出
        for line in process.stdout:
            sys.stdout.write(line)  # 在控制台显示
            f.write(line)  # 写入日志文件
            f.flush()  # 确保立即写入
        
        # 等待进程结束并获取返回码
        return_code = process.wait()
        
        # 记录实验结束时间和状态
        end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"\n实验结束时间: {end_time}\n")
        f.write(f"返回码: {return_code}\n")
        
        return return_code == 0

def run_experiments(experiments, gpu_id=None, log_dir="logs"):
    """
    运行一系列实验
    
    参数:
        experiments: 实验配置列表
        gpu_id: 指定的GPU ID，如果为None则自动选择最佳GPU
        log_dir: 日志保存目录
    """
    # 打印GPU信息
    print_gpu_info()
    
    # 如果指定了GPU，设置它
    if gpu_id is not None:
        set_gpu(gpu_id)
    
    # 创建结果目录
    results_dir = os.path.join(log_dir, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存实验配置
    with open(os.path.join(results_dir, "experiments.json"), "w") as f:
        json.dump(experiments, f, indent=2)
    
    # 运行每个实验
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n开始执行实验 {i+1}/{len(experiments)}")
        print(f"实验配置: {exp}")
        
        # 获取命令
        command = exp["command"]
        
        # 确保命令中的路径正确
        if "cd" in command and "&&" in command:
            # 分离cd命令和实际命令
            cd_part, cmd_part = command.split("&&", 1)
            cd_dir = cd_part.replace("cd", "").strip()
            
            # 检查目录是否存在
            if not os.path.exists(cd_dir):
                print(f"警告: 目录 '{cd_dir}' 不存在")
                
                # 尝试修复路径
                if "motor_dl_dl" in cd_dir:
                    fixed_dir = cd_dir.replace("motor_dl_dl", "motor_dl")
                    if os.path.exists(fixed_dir):
                        print(f"已修复路径为: '{fixed_dir}'")
                        command = f"cd {fixed_dir} && {cmd_part}"
        
        # 运行实验
        success = run_experiment(command, gpu_id, results_dir)
        
        # 记录结果
        result = {
            "experiment_id": i+1,
            "config": exp,
            "success": success,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        results.append(result)
        
        # 保存当前结果
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        if success:
            print(f"实验 {i+1} 执行成功")
        else:
            print(f"实验 {i+1} 执行失败")
        
        # 如果配置了等待时间，则等待
        if "wait_time" in exp and exp["wait_time"] > 0 and i < len(experiments) - 1:
            wait_time = exp["wait_time"]
            print(f"等待 {wait_time} 秒后开始下一个实验...")
            time.sleep(wait_time)
    
    # 打印总结
    print("\n所有实验执行完毕")
    print(f"总实验数: {len(experiments)}")
    print(f"成功: {sum(1 for r in results if r['success'])}")
    print(f"失败: {sum(1 for r in results if not r['success'])}")
    print(f"结果保存在: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行实验")
    parser.add_argument("--config", type=str, help="实验配置文件路径")
    parser.add_argument("--gpu", type=int, default=None, help="指定使用的GPU ID")
    args = parser.parse_args()
    
    if args.config:
        # 从配置文件加载实验
        with open(args.config, "r") as f:
            experiments = json.load(f)
    else:
        # 默认实验配置，减小batch_size和模型复杂度
        experiments = [
           
            
            {
                "command": "cd /root/hxq/motor_dl && python train.py --model all_model --backbone channel --distance euclidean --lr 0.00001 --feature_dim 128 --hidden_dim 16 --dropout 0.1 --n_way 4 --n_support 5 --n_query 15 --batch_size 4 --epochs 100",
                "wait_time": 10
            },
            {
                "command": "cd /root/hxq/motor_dl && python train.py --model all_model --backbone enhanced_cnn1d --distance euclidean --lr 0.00001 --feature_dim 256 --hidden_dim 32 --dropout 0.1 --n_way 4 --n_support 5 --n_query 15 --batch_size 4 --epochs 100",
                "wait_time": 10
            },
        ]
    
    # 运行实验
    run_experiments(experiments, args.gpu)