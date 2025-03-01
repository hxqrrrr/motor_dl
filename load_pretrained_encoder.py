import sys
import os
import torch
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from models.ProtoNet import ProtoNet
from utils.utils_train import get_model


def load_pretrained_encoder(model, encoder_path):
    """
    将预训练的编码器加载到ProtoNet模型中
    
    参数:
        model: ProtoNet模型实例
        encoder_path: 预训练编码器的路径
    
    返回:
        model: 加载了预训练编码器的ProtoNet模型
    """
    print(f"加载预训练编码器: {encoder_path}")
    
    # 加载预训练编码器的状态字典
    encoder_state_dict = torch.load(encoder_path, map_location='cpu')
    
    # 获取模型的状态字典
    model_state_dict = model.state_dict()
    
    # 创建一个新的状态字典，用于存储需要更新的参数
    updated_state_dict = {}
    
    # 遍历预训练编码器的参数
    for key, value in encoder_state_dict.items():
        # 构建ProtoNet中对应的参数名
        model_key = f"encoder.{key}"
        
        # 检查该参数是否存在于ProtoNet模型中
        if model_key in model_state_dict:
            # 检查参数形状是否匹配
            if model_state_dict[model_key].shape == value.shape:
                updated_state_dict[model_key] = value
                print(f"加载参数: {model_key}")
            else:
                print(f"参数形状不匹配，跳过: {model_key}")
        else:
            print(f"模型中不存在参数: {model_key}")
    
    # 更新模型的参数
    model_state_dict.update(updated_state_dict)
    model.load_state_dict(model_state_dict)
    
    print(f"成功加载预训练编码器，共加载 {len(updated_state_dict)} 个参数")
    
    return model


def save_model_with_pretrained_encoder(model, save_path):
    """
    保存加载了预训练编码器的模型
    
    参数:
        model: 加载了预训练编码器的模型
        save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将预训练编码器加载到ProtoNet模型中')
    parser.add_argument('--encoder_path', type=str, required=True,
                      help='预训练编码器的路径，例如: runs/encoder_pretrain_cnn1d_20240301_120000/best_encoder_val_acc_95.50.pth')
    parser.add_argument('--model', type=str, default='protonet',
                      choices=['protonet', 'protonet_attention', 'all_model'],
                      help='要加载编码器的模型类型')
    parser.add_argument('--in_channels', type=int, default=5, help='输入通道数')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--feature_dim', type=int, default=64, help='特征维度')
    parser.add_argument('--backbone', type=str, default='cnn1d',
                      choices=['cnn1d', 'channel', 'spatial', 'cbam'],
                      help='骨干网络类型')
    parser.add_argument('--distance_type', type=str, default='euclidean',
                      choices=['euclidean', 'cosine', 'relation', 'relation_selfattention'],
                      help='距离度量类型')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比率')
    args = parser.parse_args()
    
    # 检查预训练编码器文件是否存在
    if not os.path.exists(args.encoder_path):
        print(f"错误: 预训练编码器文件不存在: {args.encoder_path}")
        sys.exit(1)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'runs/pretrained_{args.model}_{args.backbone}_{args.distance_type}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建ProtoNet模型
    model = get_model(
        model_name=args.model,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        backbone=args.backbone,
        distance_type=args.distance_type,
        dropout=args.dropout
    )
    
    # 加载预训练编码器
    model = load_pretrained_encoder(model, args.encoder_path)
    
    # 保存加载了预训练编码器的模型
    save_path = os.path.join(save_dir, f'pretrained_{args.model}.pth')
    save_model_with_pretrained_encoder(model, save_path)
    
    print(f"\n预训练编码器已成功加载到{args.model}模型中")
    print(f"模型已保存到: {save_path}")
    print(f"\n现在您可以使用以下命令进行元学习训练:")
    print(f"python pretrain.py --model {args.model} --backbone {args.backbone} --distance {args.distance_type} --pretrained {save_path}") 