import torch
import subprocess
import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Union

def get_gpu_info() -> Dict:
    """
    获取系统中所有GPU的详细信息
    
    返回:
        Dict: 包含GPU信息的字典，格式为:
        {
            'gpu_count': 总GPU数量,
            'cuda_available': CUDA是否可用,
            'gpus': [
                {
                    'id': GPU ID,
                    'name': GPU名称,
                    'total_memory': 总内存(MB),
                    'used_memory': 已用内存(MB),
                    'free_memory': 可用内存(MB),
                    'utilization': GPU利用率(%),
                    'processes': [{'pid': 进程ID, 'name': 进程名, 'memory': 内存使用量}]
                },
                ...
            ]
        }
    """
    result = {
        'gpu_count': 0,
        'cuda_available': torch.cuda.is_available(),
        'gpus': []
    }
    
    if not result['cuda_available']:
        return result
    
    # 获取GPU数量
    result['gpu_count'] = torch.cuda.device_count()
    
    try:
        # 尝试使用nvidia-smi获取详细信息
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        
        # 获取进程信息
        process_output = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', 
             '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        
        # 获取GPU UUID映射
        uuid_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            universal_newlines=True
        )
        
        # 解析UUID映射
        uuid_map = {}
        for line in uuid_output.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) == 2:
                    idx, uuid = parts
                    uuid_map[uuid] = int(idx)
        
        # 解析进程信息
        processes_by_gpu = {}
        for line in process_output.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_uuid, pid, process_name, memory = parts
                    if gpu_uuid in uuid_map:
                        gpu_idx = uuid_map[gpu_uuid]
                        if gpu_idx not in processes_by_gpu:
                            processes_by_gpu[gpu_idx] = []
                        processes_by_gpu[gpu_idx].append({
                            'pid': int(pid),
                            'name': process_name,
                            'memory': float(memory)
                        })
        
        # 解析GPU信息
        for line in nvidia_smi_output.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    idx, name, total_mem, used_mem, free_mem, util = parts
                    gpu_info = {
                        'id': int(idx),
                        'name': name,
                        'total_memory': float(total_mem),
                        'used_memory': float(used_mem),
                        'free_memory': float(free_mem),
                        'utilization': float(util),
                        'processes': processes_by_gpu.get(int(idx), [])
                    }
                    result['gpus'].append(gpu_info)
    
    except (subprocess.SubprocessError, FileNotFoundError):
        # 如果nvidia-smi不可用，使用PyTorch获取基本信息
        for i in range(result['gpu_count']):
            gpu_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024**2),
                'used_memory': torch.cuda.memory_allocated(i) / (1024**2),
                'free_memory': (torch.cuda.get_device_properties(i).total_memory - 
                               torch.cuda.memory_allocated(i)) / (1024**2),
                'utilization': None,  # PyTorch无法获取利用率
                'processes': []  # PyTorch无法获取进程信息
            }
            result['gpus'].append(gpu_info)
    
    return result

def select_best_gpu() -> int:
    """
    选择最佳GPU（内存使用最少的GPU）
    
    返回:
        int: 最佳GPU的ID，如果没有可用GPU则返回-1
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info['cuda_available'] or gpu_info['gpu_count'] == 0:
        return -1
    
    # 按可用内存排序，选择可用内存最多的GPU
    gpus_sorted = sorted(gpu_info['gpus'], key=lambda x: x['free_memory'], reverse=True)
    return gpus_sorted[0]['id']

def print_gpu_info() -> None:
    """
    打印所有GPU的详细信息
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info['cuda_available']:
        print("系统中没有可用的CUDA设备")
        return
    
    print(f"系统中共有 {gpu_info['gpu_count']} 个GPU:")
    
    for gpu in gpu_info['gpus']:
        print(f"\nGPU {gpu['id']}: {gpu['name']}")
        print(f"  内存: {gpu['used_memory']:.0f}MB / {gpu['total_memory']:.0f}MB "
              f"(可用: {gpu['free_memory']:.0f}MB)")
        
        if gpu['utilization'] is not None:
            print(f"  利用率: {gpu['utilization']:.1f}%")
        
        if gpu['processes']:
            print("  运行中的进程:")
            for proc in gpu['processes']:
                print(f"    PID {proc['pid']}: {proc['name']} ({proc['memory']:.0f}MB)")

def set_gpu(gpu_id: Optional[int] = None) -> int:
    """
    设置要使用的GPU
    
    参数:
        gpu_id: 指定的GPU ID，如果为None则自动选择最佳GPU
        
    返回:
        int: 实际使用的GPU ID，如果使用CPU则返回-1
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        return -1
    
    if gpu_id is None:
        # 自动选择最佳GPU
        gpu_id = select_best_gpu()
        if gpu_id == -1:
            print("没有找到可用的GPU，将使用CPU")
            return -1
    
    # 检查指定的GPU ID是否有效
    gpu_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= gpu_count:
        print(f"指定的GPU ID {gpu_id} 无效，有效范围: 0-{gpu_count-1}")
        print(f"将自动选择最佳GPU")
        gpu_id = select_best_gpu()
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"已设置使用GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
    
    return gpu_id

if __name__ == "__main__":
    # 如果直接运行此脚本，打印所有GPU信息
    print_gpu_info()
    
    # 如果有命令行参数，则设置指定的GPU
    if len(sys.argv) > 1:
        try:
            gpu_id = int(sys.argv[1])
            set_gpu(gpu_id)
        except ValueError:
            print(f"无效的GPU ID: {sys.argv[1]}")
