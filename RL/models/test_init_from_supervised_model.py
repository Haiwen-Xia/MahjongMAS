import os
from models.actor import ResNet34Actor
from models.critic import ResNet34CentralizedCritic
from models.utils import load_actor_from_supervised_model, load_critic_from_actor_feature_extractor
import torch


if __name__ == "__main__":
    # 定义参数
    supervised_model_path = './supervised_model/best_model.pkl'
    input_channels = 187 # 必须与训练 Actor 时使用的特征维度一致
    output_channels = 235 # 必须与动作空间大小一致
    actor_save_path = './initial_models/actor_from_sl.pth' # 定义保存路径

    # 调用函数
    newly_loaded_actor = load_actor_from_supervised_model(
        model_path=supervised_model_path,
        in_channels=input_channels,
        # out_channels=output_channels,
        save_path=actor_save_path
    )

    if newly_loaded_actor:
        print("\n函数成功返回了一个 ResNet34Actor 实例。")
        print("检查已保存的文件是否存在:", os.path.isfile(actor_save_path))
        
        # 验证权重是否加载正确
        # 比较原始 SL 模型和一个新加载的 Actor 模型的参数
        original_weights = torch.load(supervised_model_path, map_location='cpu')
        actor_weights = torch.load(actor_save_path, map_location='cpu')
        
        # 随机选择一个键进行比较
        original_key = 'fc.weight'
        actor_key = 'resnet34.fc.weight'
        
        if torch.equal(original_weights[original_key], actor_weights[actor_key]):
            print(f"权重验证成功：'{original_key}' 和 '{actor_key}' 的张量相同。")
        else:
            print("权重验证失败！")
    else:
        print("\n函数未能成功创建和加载 Actor 模型。")

    # 定义参数
    actor_model_path_for_init = 'supervised_model/best_model.pkl'
    # Critic 的参数
    critic_obs_channels = 187 # 必须与 actor/SL 模型一致
    critic_extra_channels = 16 
    critic_save_path = './initial_models/centralized_critic_initialized.pth'

    # 调用函数来创建和初始化 Critic
    newly_initialized_critic = load_critic_from_actor_feature_extractor(
        actor_model_path=actor_model_path_for_init,
        critic_in_channels_obs=critic_obs_channels,
        critic_in_channels_extra=critic_extra_channels,
        save_path=critic_save_path
    )

    if newly_initialized_critic:
        print("\n函数成功返回了一个 ResNet34CentralizedCritic 实例。")
        print("检查已保存的文件是否存在:", os.path.isfile(critic_save_path))
        
        # 验证权重是否加载正确
        print("\n正在验证权重...")
        original_actor_weights = torch.load(actor_model_path_for_init, map_location='cpu')
        initialized_critic_weights = torch.load(critic_save_path, map_location='cpu')
        
        # 随机选择一个 feature_extractor 的参数进行比较
        original_key = 'feature_extractor.layer1.weight'
        critic_key = 'feature_extractor_obs.layer1.weight'
        
        if torch.equal(original_actor_weights[original_key], initialized_critic_weights[critic_key]):
            print(f"权重验证成功：'{original_key}' 和 '{critic_key}' 的张量相同。")
        else:
            print("权重验证失败！")
            
        # 检查 Critic 的其他部分是否保持其自己的（随机）权重
        # 例如，检查 critic_head_mlp 的权重是否存在于原始 actor 权重中
        critic_mlp_key = 'critic_head_mlp.2.weight' # MLP 最后一层的权重
        is_mlp_key_in_original = any(k.endswith(critic_mlp_key) for k in original_actor_weights.keys())
        if not is_mlp_key_in_original:
             print(f"正确：Critic 的 MLP 头部 ({critic_mlp_key}) 权重不在原始 Actor 模型中。")
        else:
            print(f"错误：Critic 的 MLP 头部权重似乎与原始 Actor 模型中的某个权重重名或被错误加载。")

    else:
        print("\n函数未能成功创建和加载 Centralized Critic 模型。")
