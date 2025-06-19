import math 
import time
from torch.utils.tensorboard.writer import SummaryWriter
import logging
import os
import datetime




# 用于动态调整学习率
def calculate_scheduled_lr(current_iter, base_lr, warmup_iters, total_iters_for_decay, min_lr, initial_lr_for_warmup):
    """
    计算带有线性预热和余弦退火的学习率。

    Args:
        current_iter (int): 当前的训练迭代次数 (从0开始)。
        base_lr (float): 预热后的基础学习率 (峰值学习率)。
        warmup_iters (int): 线性预热的迭代次数。
        total_iters_for_decay (int): 从预热结束到达到最小学习率的总衰减迭代次数。
                                    注意：这是指衰减阶段的长度，不是总训练迭代次数。
                                    总的调度长度是 warmup_iters + total_iters_for_decay。
        min_lr (float): 学习率的下限。
        initial_lr_for_warmup (float): 预热阶段开始时的学习率。

    Returns:
        float: 计算得到的当前迭代的学习率。
    """
    if warmup_iters > 0 and current_iter < warmup_iters:
        # 线性预热阶段
        # 学习率从 initial_lr_for_warmup 线性增加到 base_lr
        # 当 current_iter = 0, lr = initial_lr_for_warmup
        # 当 current_iter = warmup_iters, lr = base_lr
        # progress = current_iter / float(warmup_iters) # 这样在 warmup_iters-1 时不到 base_lr
        # 为了在第 warmup_iters 步达到 base_lr (即预热完成后的第一个decay step使用base_lr)，
        # 或者说在 warmup_iters-1 步接近 base_lr，这里采用 current_iter / warmup_iters 的比例
        lr = initial_lr_for_warmup + (base_lr - initial_lr_for_warmup) * (current_iter / float(warmup_iters))
        return lr
    else:
        # 余弦退火阶段 (或在预热之后，衰减完成之后保持min_lr)
        # 已经完成预热，或者没有预热阶段 (warmup_iters == 0)
        
        iter_after_warmup = current_iter - warmup_iters
        
        if iter_after_warmup >= total_iters_for_decay:
            # 如果已经超过了总衰减步数，则学习率保持在 min_lr
            return min_lr
        
        if total_iters_for_decay <= 0: # 防止除以零，如果衰减步数为0或负，则直接使用min_lr (或base_lr if no decay intended)
            return min_lr # 或者 base_lr，取决于期望行为

        # 计算在衰减阶段的进度 (0.0 to 1.0)
        progress = iter_after_warmup / float(total_iters_for_decay)
        
        # 余弦退火公式: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        cosine_component = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine_component
        return lr
    


# 用来构建 logging 和 TensorBoard 的函数
# 在主进程以及每个 learner 和 actor 进程中都会被使用

def create_experiment_dir_structure(base_log_dir, experiment_config):
    """
    创建新的分层实验目录结构
    
    Args:
        base_log_dir (str): 基础日志目录 (e.g., "logs")
        experiment_config (dict): 实验配置，包含用于生成目录名的参数
    
    Returns:
        tuple: (main_experiment_dir, additional_experiment_dir)
    """
    # 生成时间戳目录 (e.g., "2025-06-19_PM")
    now = datetime.datetime.now()
    time_dir = now.strftime("%Y-%m-%d_%p")  # PM/AM
    
    # 生成实验配置目录名 (e.g., "lr3e5_bs256_na1_19-00")
    lr = experiment_config.get('lr_actor', 3e-5)
    batch_size = experiment_config.get('batch_size', 256)
    num_actors = experiment_config.get('num_actors', 1)
    
    
    config_dir = f"lr{lr:.0e}_bs{batch_size}_na{num_actors}_{now.strftime("%H-%M")}"
    
    # 创建主要和附加目录
    main_experiment_dir = os.path.join(base_log_dir, time_dir, config_dir)
    additional_experiment_dir = os.path.join(base_log_dir, f"{time_dir}_additional", config_dir)
    
    # 创建子目录结构
    subdirs_main = ['logs', 'tensorboard', 'checkpoints']
    subdirs_additional = ['tensorboard_full']
    
    for subdir in subdirs_main:
        os.makedirs(os.path.join(main_experiment_dir, subdir), exist_ok=True)
    
    for subdir in subdirs_additional:
        os.makedirs(os.path.join(additional_experiment_dir, subdir), exist_ok=True)
    
    return main_experiment_dir, additional_experiment_dir

def setup_process_logging_and_tensorboard(base_log_dir, experiment_config, process_name, 
                                         log_type='main', log_level=logging.INFO):
    """
    为多进程应用中的特定进程初始化日志记录和TensorBoard。
    
    Args:
        base_log_dir (str): 基础日志目录
        experiment_config (dict): 实验配置字典
        process_name (str): 进程名称标识 (e.g., 'main_train', 'learner', 'actor', 'inference_server')
        log_type (str): 日志类型 ('main' 为主要日志, 'detailed' 为详细日志)
        log_level (int, optional): 日志级别. 默认为 logging.INFO.
    
    Returns:
        tuple: 包含以下内容的元组:
            - logger (logging.Logger): 配置好的logger实例
            - writer (torch.utils.tensorboard.SummaryWriter): 配置好的TensorBoard writer
            - log_paths (dict): 包含各种路径信息的字典
    """
    # 创建实验目录结构
    main_exp_dir, additional_exp_dir = create_experiment_dir_structure(base_log_dir, experiment_config)
    
    # 根据log_type选择目标目录
    if log_type == 'main':
        target_dir = main_exp_dir
        tb_subdir = 'tensorboard'
        log_subdir = 'logs'
    else:  # detailed
        target_dir = additional_exp_dir
        tb_subdir = 'tensorboard_full'
        log_subdir = '.'  # 直接在additional目录下
    
    # --- 1. 构建唯一的名称和路径 ---
    logger_name = f"{process_name}_{log_type}"
    
    # 根据进程类型选择日志文件名
    if log_type == 'main':
        if process_name == 'main_train':
            log_file_name = 'unified.log'
        elif process_name == 'learner':
            log_file_name = 'training.log'
        elif process_name.startswith('evaluator'):
            log_file_name = 'evaluation.log'
        else:
            log_file_name = f'{process_name}.log'
    else:  # detailed
        log_file_name = f'{process_name}.log'
    
    # --- 2. 设置文件日志 ---
    log_dir = os.path.join(target_dir, log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # 统一主日志文件路径
    unified_log_path = os.path.join(main_exp_dir, 'logs', 'unified.log')
    
    # 获取此特定进程的logger实例
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # 防止消息传递给root logger的处理器
    
    # 移除现有处理器以避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建进程专用文件处理器
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 使用追加模式
        if log_type == 'main':
            # 主要日志使用简洁格式
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        else:
            # 详细日志使用完整格式
            formatter = logging.Formatter(
                '%(asctime)s [%(processName)s:%(process)d/%(levelname)s] %(name)s: %(message)s'
            )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # 回退到控制台输出
        print(f"Error setting up file logger for {logger_name}: {e}. Logging to console for this logger.")
        stream_handler_fallback = logging.StreamHandler()
        formatter_fallback = logging.Formatter(
            '%(asctime)s - %(name)s - [%(processName)s:%(process)d] - %(levelname)s - %(message)s'
        )
        stream_handler_fallback.setFormatter(formatter_fallback)
        logger.addHandler(stream_handler_fallback)
    
    # --- 新增：为所有非main_train进程添加统一日志处理器 ---
    if process_name != 'main_train':
        try:
            # 确保统一日志目录存在
            os.makedirs(os.path.dirname(unified_log_path), exist_ok=True)
            
            # 创建统一日志处理器（支持多进程安全写入）
            unified_handler = logging.FileHandler(unified_log_path, mode='a', encoding='utf-8')
            
            # 为统一日志使用带进程标识的格式
            unified_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s|%(processName)s] %(message)s'
            )
            unified_handler.setFormatter(unified_formatter)
            
            # 只记录INFO及以上级别的重要信息到统一日志
            unified_handler.setLevel(logging.INFO)
            logger.addHandler(unified_handler)
            
        except Exception as e:
            print(f"Warning: Failed to setup unified logging for {process_name}: {e}")
    
    # --- 3. 设置TensorBoard ---
    tensorboard_log_path = os.path.join(target_dir, tb_subdir, process_name)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    
    try:
        writer = SummaryWriter(log_dir=tensorboard_log_path)
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoard SummaryWriter for {logger_name} at {tensorboard_log_path}: {e}")
        writer = None
    
    # 构建路径信息字典
    log_paths = {
        'main_experiment_dir': main_exp_dir,
        'additional_experiment_dir': additional_exp_dir,
        'log_file_path': log_file_path,
        'unified_log_path': unified_log_path,  # 新增统一日志路径
        'tensorboard_path': tensorboard_log_path,
        'config_save_path': os.path.join(main_exp_dir, 'config.json'),
        'checkpoint_dir': os.path.join(main_exp_dir, 'checkpoints')
    }
    
    logger.info(f"File logging for '{logger_name}' initialized. Log file: {log_file_path}")
    if process_name != 'main_train':
        logger.info(f"Unified logging enabled. Key messages will also be written to: {unified_log_path}")
    if writer:
        logger.info(f"TensorBoard logging for '{logger_name}' initialized. Log directory: {tensorboard_log_path}")
    
    # 打印到终端和unified.log (如果这是main进程)
    if process_name == 'main_train':
        print(f"=== Experiment Directory Structure ===")
        print(f"Main experiment dir: {main_exp_dir}")
        print(f"Additional logs dir: {additional_exp_dir}")
        print(f"Unified log file: {unified_log_path}")
        print(f"Config will be saved to: {log_paths['config_save_path']}")
        print(f"Checkpoints dir: {log_paths['checkpoint_dir']}")
        print(f"=====================================")
    
    return logger, writer, log_paths

def save_experiment_config(config, config_save_path):
    """
    保存实验配置到指定路径
    
    Args:
        config (dict): 实验配置字典
        config_save_path (str): 配置文件保存路径
    """
    try:
        import json
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str, ensure_ascii=False)
        print(f"Experiment config saved to: {config_save_path}")
    except Exception as e:
        print(f"Failed to save experiment config: {e}")

# 为了向后兼容，保留原函数签名
def setup_process_logging_and_tensorboard_legacy(base_log_dir, run_name, process_name, log_level=logging.INFO):
    """向后兼容的函数，调用新的日志设置函数"""
    # 构建临时配置
    temp_config = {
        'experiment_name': run_name,
        'lr_actor': 3e-5,
        'batch_size': 256,
        'num_actors': 1
    }
    logger, writer, _ = setup_process_logging_and_tensorboard(
        base_log_dir, temp_config, process_name, 'main', log_level
    )
    return logger, writer