
import math 
import time
from torch.utils.tensorboard import SummaryWriter
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

def setup_process_logging_and_tensorboard(base_log_dir, run_name, process_name, log_level=logging.INFO):
    """
    Initializes logging and TensorBoard for a specific process in a multi-process application.

    Args:
        base_log_dir (str): The base directory where all logs (file logs and TensorBoard) for the run will be stored.
        run_name (str): The name of the current experiment run.
        process_name (str): A string identifying the name of the process (e.g., 'main', 'learner', 'actor').
        log_level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Returns:
        tuple: A tuple containing:
            - logger (logging.Logger): The configured logger instance for this process.
            - writer (torch.utils.tensorboard.SummaryWriter): The configured TensorBoard writer for this process.
    """
    # --- 1. Construct unique names and paths ---
    logger_name = process_name
    log_file_name = f"{process_name}.log"
    tb_subdir_name = process_name

    # --- 2. Setup File Logging ---
    # Directory for file logs of this specific run
    run_file_log_dir = os.path.join(base_log_dir, "file_logs", run_name)
    os.makedirs(run_file_log_dir, exist_ok=True)
    log_file_path = os.path.join(run_file_log_dir, log_file_name)

    # Get a logger instance for this specific process
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False # Prevent messages from being passed to the root logger's handlers

    # Remove any existing handlers to avoid duplication if this function is called multiple times for the same logger name
    # (though ideally, it's called once per process for a unique logger_name)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite for this specific process log
        formatter = logging.Formatter(
            '%(asctime)s [%(processName)s/%(levelname)s] %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to console if file handler fails
        print(f"Error setting up file logger for {logger_name}: {e}. Logging to console for this logger.")
        # BasicConfig might have already been called by another process or the main thread.
        # Adding a stream handler directly to this logger is safer.
        stream_handler_fallback = logging.StreamHandler()
        formatter_fallback = logging.Formatter(
             '%(asctime)s - %(name)s - [%(processName)s:%(process)d] - %(levelname)s - %(message)s'
        )
        stream_handler_fallback.setFormatter(formatter_fallback)
        logger.addHandler(stream_handler_fallback)


    # --- 3. Setup TensorBoard ---
    # Central TensorBoard directory for all runs, then specific run, then specific process type
    tensorboard_log_path = os.path.join(base_log_dir, "tensorboard_logs", run_name, tb_subdir_name)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    
    try:
        writer = SummaryWriter(log_dir=tensorboard_log_path)
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoard SummaryWriter for {logger_name} at {tensorboard_log_path}: {e}")
        writer = None # Or a dummy writer that does nothing

    logger.info(f"File logging for '{logger_name}' initialized. Log file: {log_file_path}")
    if writer:
        logger.info(f"TensorBoard logging for '{logger_name}' initialized. Log directory: {tensorboard_log_path}")

    return logger, writer