import numpy as np
from typing import Dict, List, Optional, Union, Any
import json
import os

class LabelEncoder:
    """
    将分类标签转换为数值表示的编码器
    
    功能:
    1. 将字符串或其他类型的标签转换为连续的整数索引
    2. 支持保存和加载编码映射
    3. 支持未知标签处理
    4. 提供反向转换功能
    
    示例:
    ```python
    # 初始化编码器
    encoder = LabelEncoder()
    
    # 拟合标签
    labels = ['cat', 'dog', 'bird', 'cat', 'dog']
    encoder.fit(labels)
    
    # 转换标签为数值
    encoded = encoder.transform(labels)  # [0, 1, 2, 0, 1]
    
    # 拟合并转换
    encoded = encoder.fit_transform(labels)  # [0, 1, 2, 0, 1]
    
    # 反向转换
    original = encoder.inverse_transform([0, 1, 2])  # ['cat', 'dog', 'bird']
    
    # 保存编码映射
    encoder.save('label_mapping.json')
    
    # 加载编码映射
    encoder.load('label_mapping.json')
    ```
    """
    
    def __init__(self, unknown_label: int = -1):
        """
        初始化LabelEncoder
        
        参数:
            unknown_label: 用于表示未知标签的值，默认为-1
        """
        self.classes_ = []  # 存储唯一标签
        self.label_to_index = {}  # 标签到索引的映射
        self.index_to_label = {}  # 索引到标签的映射
        self.unknown_label = unknown_label
        self.is_fitted = False
    
    def fit(self, y: List[Any]) -> 'LabelEncoder':
        """
        拟合编码器，建立标签到索引的映射
        
        参数:
            y: 标签列表
            
        返回:
            self: 返回自身实例
        """
        unique_labels = sorted(set(y), key=lambda x: str(x))
        self.classes_ = unique_labels
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        self.is_fitted = True
        return self
    
    def transform(self, y: List[Any]) -> np.ndarray:
        """
        将标签转换为数值索引
        
        参数:
            y: 标签列表
            
        返回:
            encoded: 编码后的数值索引数组
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，请先调用fit方法")
        
        encoded = np.array([
            self.label_to_index.get(label, self.unknown_label) 
            for label in y
        ])
        return encoded
    
    def fit_transform(self, y: List[Any]) -> np.ndarray:
        """
        拟合编码器并转换标签
        
        参数:
            y: 标签列表
            
        返回:
            encoded: 编码后的数值索引数组
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: Union[List[int], np.ndarray]) -> List[Any]:
        """
        将数值索引转换回原始标签
        
        参数:
            y: 数值索引列表或数组
            
        返回:
            original: 原始标签列表
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，请先调用fit方法")
        
        original = [
            self.index_to_label.get(idx, None) 
            for idx in y
        ]
        return original
    
    def save(self, filepath: str) -> None:
        """
        保存编码映射到JSON文件
        
        参数:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未拟合，无法保存")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 保存映射
        mapping = {
            "classes": self.classes_,
            "label_to_index": self.label_to_index,
            "unknown_label": self.unknown_label
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> 'LabelEncoder':
        """
        从JSON文件加载编码映射
        
        参数:
            filepath: 文件路径
            
        返回:
            self: 返回自身实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        self.classes_ = mapping["classes"]
        self.label_to_index = {str(k): v for k, v in mapping["label_to_index"].items()}
        self.unknown_label = mapping["unknown_label"]
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.is_fitted = True
        
        return self
    
    def __len__(self) -> int:
        """返回类别数量"""
        return len(self.classes_)
    
    def __repr__(self) -> str:
        """返回编码器的字符串表示"""
        if self.is_fitted:
            return f"LabelEncoder(classes={self.classes_}, unknown_label={self.unknown_label})"
        else:
            return f"LabelEncoder(unknown_label={self.unknown_label}, not fitted)" 