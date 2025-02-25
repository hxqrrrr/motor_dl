import subprocess
import time
import os
from typing import List, Dict
import json
from datetime import datetime

def run_command(command: str, log_file: str = None) -> bool:
    """
    执行单个命令并记录输出
    
    参数:
        command: 要执行的命令
        log_file: 日志文件路径
    
    返回:
        bool: 命令是否成功执行
    """
    try:
        # 如果指定了日志文件，将输出重定向到文件
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"执行命令: {command}\n")
                f.write(f"开始时间: {datetime.now()}\n")
                f.write(f"{'='*50}\n\n")
                
            with open(log_file, 'a', encoding='utf-8') as f:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # 实时输出命令执行结果
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    
                process.wait()
                
                f.write(f"\n{'='*50}\n")
                f.write(f"命令执行完成时间: {datetime.now()}\n")
                f.write(f"返回码: {process.returncode}\n")
                f.write(f"{'='*50}\n\n")
                
        else:
            # 如果没有指定日志文件，直接执行命令
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
        return process.returncode == 0
        
    except Exception as e:
        print(f"执行命令时出错: {str(e)}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n错误: {str(e)}\n")
        return False

def run_experiments(experiments: List[Dict[str, str]], log_dir: str = "logs") -> None:
    """
    连续执行多个实验
    
    参数:
        experiments: 实验配置列表
        log_dir: 日志目录
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间作为实验组的标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建本次实验的日志文件
    log_file = os.path.join(log_dir, f"experiment_group_{timestamp}.log")
    
    # 记录实验开始
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"实验组开始时间: {datetime.now()}\n")
        f.write(f"实验配置:\n{json.dumps(experiments, indent=2, ensure_ascii=False)}\n")
        f.write(f"{'='*50}\n\n")
    
    # 执行每个实验
    for i, exp in enumerate(experiments, 1):
        print(f"\n开始执行实验 {i}/{len(experiments)}")
        print(f"实验配置: {exp}")
        
        # 构建命令
        command = exp['command']
        
        # 执行命令
        success = run_command(command, log_file)
        
        if not success:
            print(f"实验 {i} 执行失败")
            # 可以选择在这里中断所有实验
            # break
        
        # 实验之间添加间隔时间（可选）
        if i < len(experiments):
            time.sleep(2)
    
    # 记录实验组结束
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"实验组结束时间: {datetime.now()}\n")

if __name__ == "__main__":
    # Python解释器的完整路径
    python_path = "C:/Users/hxq11/anaconda3/envs/CLIP-LoRA/python.exe"
    tensorboard_path = "C:/Users/hxq11/anaconda3/envs/CLIP-LoRA/Scripts/tensorboard.exe"
    
    # 定义要执行的实验列表
    experiments_benchmark = [
      
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_cbam_cosine_20250224_172035/best_model_val_acc_0.8530.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_cbam_euclidean_20250224_205957/best_model_val_acc_0.8616.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_channel_cosine_20250224_162136/best_model_val_acc_0.7894.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_channel_euclidean_20250224_185049/best_model_val_acc_0.8546.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_cnn1d_cosine_20250224_150113/best_model_val_acc_0.8185.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_cnn1d_cosine_20250224_153348/best_model_val_acc_0.8632.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_cnn1d_euclidean_20250224_181335/best_model_val_acc_0.8694.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/all_model_spatial_cosine_20250224_164030/best_model_val_acc_0.8272.pth"'
        },
        {
            "command": f'"{python_path}" benchmark.py --model_name all_model --model_path "runs/protonet_20250219_143734/best_model_0.8965.pth"'
        }
    ]
    experiments_train = [
        {
            "command": f'"{python_path}" train.py --model all_model --backbone cnn1d --distance_type euclidean '
                      f'--in_channels 5 --hidden_dim 128 --feature_dim 64 --dropout 0.3 '
                      f'--scheduler cosine --warmup_epochs 5 --warmup_start_lr 1e-6 --lr 0.001'
        },
        {
            "command": f'"{python_path}" train.py --model all_model --backbone cnn1d --distance_type cosine '
                      f'--in_channels 5 --hidden_dim 128 --feature_dim 256 --dropout 0.3 '
                      f'--scheduler cosine --warmup_epochs 5 --warmup_start_lr 1e-6 --lr 0.001'
        },
        {
            "command": f'"{python_path}" train.py --model all_model --backbone cbam --distance_type euclidean '
                      f'--in_channels 5 --hidden_dim 256 --feature_dim 128 --dropout 0.3 '
                      f'--scheduler cosine --warmup_epochs 5 --warmup_start_lr 1e-6 --lr 0.001'
        },
        {
            "command": f'"{python_path}" train.py --model all_model --backbone cbam --distance_type euclidean '
                      f'--in_channels 5 --hidden_dim 256 --feature_dim 256 --dropout 0.3 '
                      f'--scheduler cosine --warmup_epochs 5 --warmup_start_lr 1e-6 --lr 0.001'
        },
        
    ]
    # 执行实验
    # run_experiments(experiments_benchmark) 
    run_experiments(experiments_train)