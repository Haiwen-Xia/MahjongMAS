import torch
import torch.multiprocessing as mp # Use torch multiprocessing for better CUDA handling if needed
import os
import logging
import json

# Assume these imports are correct relative to your project structure
from replay_buffer import ReplayBuffer 
from actor import Actor
from learner import Learner

from utils import setup_process_logging_and_tensorboard

# --- Configuration ---
# It's often better to load config from a file (e.g., YAML, JSON) or use argparse
CONFIG = {
    # Experiment Meta
    'experiment_name': "Freeze_Critic_FE_Entropy=-1e-4_model_pool_size=200", # Use underscores or avoid special chars for dir names
    'log_base_dir': '/home/dataset-assist-0/data/Mahjong/RL/log', # Base directory for logs and TensorBoard
    'checkpoint_base_dir': '/home/dataset-assist-0/data/Mahjong/RL/model', # Base directory for checkpoints

    # Data & Replay Buffer
    'replay_buffer_size': 50000,
    'replay_buffer_episode_capacity': 400, # Renamed for clarity
    'min_sample_to_start_learner': 20000, # Increased wait time based on batch size? Renamed.

    # Model & Training
    'supervised_model_path': '/home/dataset-assist-0/data/Mahjong/RL/supervised_model/best_model.pkl',
    'in_channels': 187,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Distributed Setup
    'model_pool_size': 200,
    'model_pool_name': 'model-pool-v2', # Example name
    'num_actors': 8,
    'episodes_per_actor': 25000, # How many episodes each actor runs before exiting

    # Learner Hyperparameters
    'batch_size': 1024, # Increased batch size
    'epochs_per_batch': 5, # Renamed 'epochs' for clarity (PPO inner loops)

    ## Learning Rate Scheduler
    # 'lr': 1e-4,
    'lr_critic_head': 3e-5,
    'lr_critic_feature_extractor': 3e-5,
    'lr_actor_head_finetune': 3e-5,
    'lr_actor_feature_extractor_finetune': 3e-5,
    
    'unfreeze_actor_head_after_iters': 500,
    'unfreeze_actor_feature_extractor_after_iters': 500,

    'use_lr_scheduler': True,
    "warmup_iterations": 250,
    'total_iterations_for_lr_decay': 500000,
    'initial_lr_warmup_critic': 1e-6,
    'min_lr_critic_schedule': 1e-6, # Minimum learning rate for scheduler

    'gamma': 0.98,      # Discount factor for GAE/TD Target
    'lambda': 0.97,     # Lambda for GAE
    'clip': 0.2,        # PPO clip epsilon
    'grad_clip_norm': 0.3,
    'value_coeff': 0.5, # Coefficient for value loss (common to scale down)
    'entropy_coeff': 1e-4,# Coefficient for entropy bonus
    'ckpt_save_interval_seconds': 600, # Save checkpoint every N seconds (e.g., 10 minutes)
    'filter_single_action_steps': False, # 是否过滤掉只有单个可能 action 的时间步

    # 多样化 model_pool
    'p_opponent_historical' : 0.2,
    'opponent_sampling_k' : 196, 
    'opponent_model_change_interval': 1, # 每多少个 episode 替换一次对手
    'actor_model_change_interval': 1,


    # Added for potential use in Learner/Actor logging/config
    'log_interval_learner': 100, # Log Learner stats every N iterations
    'model_push_interval': 10   # Push model every N Learner iterations
}


def main():
    """Main function to orchestrate training."""
    
    # Generate a unique run name if needed, or use the one from config
    run_name = CONFIG['experiment_name'] # Or add timestamp: f"{CONFIG['experiment_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_base_dir = CONFIG['log_base_dir']

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [ROOT/%(levelname)s] %(message)s', # Differentiate root logs
        handlers=[logging.StreamHandler()] # Root logs to console
    )
    # --- Setup ---
    logger, writer = setup_process_logging_and_tensorboard(
        log_base_dir, run_name, process_name='main'
    )
    
    # Log configuration
    logger.info("="*50)
    logger.info(f"Starting Experiment: {run_name}")
    logger.info("Configuration:")
    logger.info(json.dumps(CONFIG, indent=4))
    logger.info("="*50)

    # --- Initialization ---
    logger.info("Initializing Replay Buffer...")
    # Pass the full config, ReplayBuffer can extract what it needs
    replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'], CONFIG['replay_buffer_episode_capacity']) 
    logger.info(f"Replay Buffer initialized with size {CONFIG['replay_buffer_size']} and episode capacity {CONFIG['replay_buffer_episode_capacity']}.")

    # Prepare checkpoint directory
    checkpoint_dir = os.path.join(CONFIG['checkpoint_base_dir'], run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    CONFIG['ckpt_save_path'] = checkpoint_dir # Update config with specific run's checkpoint path

    logger.info("Initializing Learner...")
    # Pass the full config, Learner can extract what it needs
    learner = Learner(CONFIG, replay_buffer) 
    learner.name = 'Learner' # Set process name for logging
    logger.info("Learner initialized.")

    logger.info(f"Initializing {CONFIG['num_actors']} Actors...")
    actors = []
    for i in range(CONFIG['num_actors']):
        actor_config = CONFIG.copy() # Create a copy for each actor if needed
        actor_config['name'] = f'Actor-{i}' # Assign unique name
        actor = Actor(actor_config, replay_buffer)
        actor.name = actor_config['name'] # Set process name for logging
        actors.append(actor)
    logger.info(f"{CONFIG['num_actors']} Actors initialized.")

    # --- Start Processes ---
    try:
        logger.info("Starting Learner process...")
        learner.start()
        logger.info("Starting Actor processes...")
        for actor in actors:
            actor.start()

        # --- Wait for Processes ---
        # Actors might finish after completing their episodes
        logger.info("Waiting for Actor processes to complete...")
        for actor in actors:
            actor.join() # Wait for each actor process to finish
        logger.info("All Actor processes have finished.")

        # --- Shutdown ---
        logger.info("Terminating Learner process...")
        # Sending a signal or using a queue for graceful shutdown is better
        # but terminate() is simpler for now. Learner might not save final checkpoint.
        if learner.is_alive():
             learner.terminate() # Forcefully terminate learner
             learner.join() # Wait for termination
        logger.info("Learner process terminated.")

        # Log final buffer stats if possible (ReplayBuffer needs a method)
        # logger.info(f"Final Replay Buffer size: {replay_buffer.size()}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        # Attempt graceful shutdown
        print("Attempting graceful shutdown...")
        if learner.is_alive():
            learner.terminate()
            learner.join()
        for actor in actors:
            if actor.is_alive():
                actor.terminate()
                actor.join()
        print("Processes terminated.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Terminate processes on error
        if learner.is_alive(): learner.terminate()
        for actor in actors:
           if actor.is_alive(): actor.terminate()
    finally:
        # --- Cleanup ---
        logger.info("Closing TensorBoard writer.")
        writer.close()
        logger.info("Training script finished.")
        # Close log file handlers if necessary (basicConfig usually handles this)


if __name__ == '__main__':
    # Set start method for multiprocessing if needed (e.g., 'spawn' might be more stable on some systems)
    mp.set_start_method('spawn', force=True) 
    main()