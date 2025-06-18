from models.actor import ResNet34Actor
from models.base_model import ResNet34
from models.critic import ResNet34CentralizedCritic
import os
import torch
from torch import nn
from collections import OrderedDict
import os

def load_actor_from_supervised_model(model_path, in_channels, save_path = None):
    """
    Load an actor model from a supervised learning model.
    
    Args:
        model_path (str): Path to the supervised learning model.
        in_channels (int): Number of input channels for the actor model.
        save_path (str, optional): Path to save the actor model. If None, the model will not be saved.
        
    Returns:
        ResNet34Actor: An instance of the ResNet34Actor initialized with the weights from the supervised model.
    """
    print(f"--- 开始从监督模型转换 Actor ---")
    print(f"源模型路径: {model_path}")

    # 1. 检查源模型文件是否存在
    if not os.path.isfile(model_path):
        print(f"错误: 源模型文件不存在: {model_path}")
        return None

    # 2. 加载监督学习模型的 state_dict
    #    使用 map_location='cpu' 确保即使模型是在GPU上保存的，也能在CPU上加载，增加了代码的可移植性。
    try:
        print(f"正在从 '{model_path}' 加载监督模型权重...")
        supervised_checkpoint = torch.load(model_path, map_location='cpu')

        # 检查加载的是否是 checkpoint 字典 (包含 'model_state_dict') 或直接就是 state_dict
        if isinstance(supervised_checkpoint, dict) and 'model_state_dict' in supervised_checkpoint:
            supervised_state_dict = supervised_checkpoint['model_state_dict']
            print("从 checkpoint 中提取了 'model_state_dict'。")
        elif isinstance(supervised_checkpoint, dict):
            supervised_state_dict = supervised_checkpoint
            print("加载的对象是一个 state_dict。")
        else:
            print(f"错误: 加载的文件不是预期的 state_dict 或 checkpoint 字典。类型为: {type(supervised_checkpoint)}")
            return None
    except Exception as e:
        print(f"错误: 加载监督模型文件时出错: {e}")
        return None

    # 3. 创建一个新的 ResNet34Actor 实例
    print(f"正在创建新的 ResNet34Actor 实例 (in_channels={in_channels}")
    actor_model = ResNet34Actor(in_channels=in_channels, out_channels=235)
    
    # 4. 创建一个新的 state_dict，将监督模型的键名映射到 Actor 模型的键名
    #    源键名 (SL模型): "feature_extractor.xxx", "fc.xxx"
    #    目标键名 (Actor模型): "resnet34.feature_extractor.xxx", "resnet34.fc.xxx"
    actor_state_dict_to_load = OrderedDict()
    keys_mapped_count = 0
    print("正在映射监督模型的权重到 Actor 的结构...")
    for key, param in supervised_state_dict.items():
        # 简单地在每个键前加上 "resnet34." 前缀
        new_key = f"resnet34.{key}"
        # 检查新的键名是否在 actor_model 的 state_dict 中，这是一个好的健壮性检查
        if new_key in actor_model.state_dict():
            actor_state_dict_to_load[new_key] = param
            keys_mapped_count += 1
        else:
            print(f"  > 警告: 监督模型中的键 '{key}' 映射为 '{new_key}' 后，在目标 Actor 模型中未找到。此权重将被忽略。")
    
    if keys_mapped_count == 0:
        print("错误: 未能从监督模型映射任何权重。请检查模型结构和键名是否匹配。")
        return None
        
    print(f"共映射了 {keys_mapped_count} 个权重张量。")

    # 5. 将新的 state_dict 加载到 Actor 模型中
    #    使用 strict=False，因为它允许只加载部分匹配的权重。
    #    如果 actor_state_dict_to_load 包含了 actor_model.resnet34 的所有权重，也可以用 strict=True 来确保完全匹配。
    print("正在将映射后的权重加载到 Actor 模型中...")
    try:
        actor_model.load_state_dict(actor_state_dict_to_load, strict=False)
        print("权重加载成功！")
    except Exception as e:
        print(f"错误: 将权重加载到 Actor 模型时失败: {e}")
        return None
        
    # 6. (可选) 保存新的 Actor 模型
    if save_path:
        print(f"正在将转换后的 Actor 模型保存到: {save_path}")
        try:
            # 创建保存路径的目录（如果不存在）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 直接调用 actor_model 上我们实现的 save 方法，或者直接用 torch.save
            actor_model.save(save_path)
            # 或者 torch.save(actor_model.state_dict(), save_path)
            print("Actor 模型保存成功。")
        except Exception as e:
            print(f"错误: 保存 Actor 模型失败: {e}")
            # 即使保存失败，仍然返回加载好的模型实例

    print("--- Actor 模型转换完成 ---")
    return actor_model

def load_critic_from_actor_feature_extractor(
    actor_model_path: str, 
    critic_in_channels_obs: int, 
    critic_in_channels_extra: int,
    save_path: str = None
) -> ResNet34CentralizedCritic:
    """
    创建一个 Centralized Critic，并用一个预训练好的 Actor/SL 模型 (4) 的特征提取器部分来初始化它。

    Args:
        actor_model_path (str): 预训练的 Actor 或 SL 模型 (.pth 或 .pkl) 的路径。
        critic_in_channels_obs (int): Critic 用于处理局部观测的输入通道数 (应与 Actor/SL 模型一致)。
        critic_in_channels_extra (int): Critic 用于处理额外全局信息的输入通道数。
        save_path (str, optional): 保存部分初始化的 Critic 模型权重的路径。如果为 None，则不保存。

    Returns:
        ResNet34CentralizedCritic: 一个 ResNet34CentralizedCritic 的实例，其 feature_extractor_obs 
                                   部分的权重已从指定模型加载，其余部分为随机初始化。
                                   如果加载失败，则返回 None。
    """
    print(f"--- 开始从 Actor/SL 模型初始化 Centralized Critic ---")
    print(f"源模型路径: {actor_model_path}")

    # 1. 检查源模型文件是否存在
    if not os.path.isfile(actor_model_path):
        print(f"错误: 源模型文件不存在: {actor_model_path}")
        return None

    # 2. 加载源模型的 state_dict
    try:
        print(f"正在从 '{actor_model_path}' 加载源模型权重...")
        source_checkpoint = torch.load(actor_model_path, map_location='cpu')
        
        if isinstance(source_checkpoint, dict) and 'model_state_dict' in source_checkpoint:
            source_state_dict = source_checkpoint['model_state_dict']
        elif isinstance(source_checkpoint, dict):
            source_state_dict = source_checkpoint
        else:
            print(f"错误: 加载的文件不是预期的 state_dict 或 checkpoint 字典。类型: {type(source_checkpoint)}")
            return None
    except Exception as e:
        print(f"错误: 加载源模型文件时出错: {e}")
        return None

    # 3. 创建一个新的 ResNet34CentralizedCritic 实例
    try:
        critic_model = ResNet34CentralizedCritic(
            in_channels_obs=critic_in_channels_obs,
            in_channels_extra=critic_in_channels_extra
        )
    except Exception as e:
        print(f"错误: 创建 ResNet34CentralizedCritic 实例失败: {e}")
        return None

    # 4. 提取源模型中的 feature_extractor 权重，并映射到 Critic 的 feature_extractor_obs
    critic_obs_fe_state_dict = OrderedDict()
    keys_mapped_count = 0
    print("正在从源模型中提取并映射特征提取器的权重...")
    for key, param in source_state_dict.items():
        if key.startswith('feature_extractor.'):
            # 将 "feature_extractor." 前缀替换为 "feature_extractor_obs."
            new_key = key.replace('feature_extractor.', 'feature_extractor_obs.', 1)
            
            if new_key in critic_model.state_dict():
                # 检查形状是否匹配，以增加健壮性
                if critic_model.state_dict()[new_key].shape == param.shape:
                    critic_obs_fe_state_dict[new_key] = param
                    keys_mapped_count += 1
                else:
                     print(f"  > 警告: 形状不匹配，跳过权重 '{key}' -> '{new_key}'。")
            else:
                print(f"  > 警告: 目标键 '{new_key}' 在 Critic 模型中未找到。此权重将被忽略。")

    if keys_mapped_count == 0:
        print("错误: 未能从源模型映射任何特征提取器的权重。请检查源模型是否包含 'feature_extractor.' 前缀的层。")
        return None
        
    print(f"共映射了 {keys_mapped_count} 个特征提取器权重张量。")

    # 5. 将提取出的权重加载到新的 Critic 模型中
    #    使用 strict=False，因为我们只加载了 feature_extractor_obs 部分的权重
    print("正在将权重加载到 Critic 的 feature_extractor_obs 部分...")
    try:
        critic_model.load_state_dict(critic_obs_fe_state_dict, strict=False)
        print("权重加载成功！其余部分（extra_feature_extractor, mlp_head）将保持随机初始化。")
    except Exception as e:
        print(f"错误: 将权重加载到 Critic 模型时失败: {e}")
        return None
        
    # 6. (可选) 保存部分初始化的 Critic 模型
    if save_path:
        print(f"正在将部分初始化的 Critic 模型保存到: {save_path}")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            critic_model.save(save_path) # 使用模型上定义的 save 方法
        except Exception as e:
            print(f"错误: 保存 Critic 模型失败: {e}")

    print("--- Centralized Critic 初始化完成 ---")
    return critic_model
