# Agent part
# 导入特征处理模块, 用于将游戏状态转换为模型可以理解的特征向量
from feature import FeatureAgent

# Model part
# 导入神经网络模型定义, 因为我们需要实例化网络并加载模型的权重
from model import Net

# Botzone interaction
import numpy as np
import torch

# 定义一个函数, 将观测值（游戏状态）通过模型转换为响应（动作）
def obs2response(model, obs):
    """
    根据当前观测到的游戏状态 (obs), 使用训练好的模型 (model) 来决定下一步的动作。

    Args:
        model (Net): 训练好的神经网络模型。
        obs (dict): 包含 'observation' (特征向量) 和 'action_mask' (合法动作掩码) 的字典。

    Returns:
        str: AI决策的动作字符串, 例如 "PLAY W1", "HU", "PENG" 等。
    """
    # 将观测数据和动作掩码转换为PyTorch张量, 并增加一个批次维度 (batch_size=1)
    # 'is_training': False 表示模型处于推理模式, 而非训练模式
    observation_tensor = torch.from_numpy(np.expand_dims(obs['observation'], 0))
    action_mask_tensor = torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
    
    # 将数据输入模型, 获取模型输出的 logits (每个动作的原始分数)
    logits = model({'is_training': False, 'obs': {'observation': observation_tensor, 'action_mask': action_mask_tensor}})
    
    # 从计算图中分离logits, 转换为Numpy数组, 展平, 并找到分数最高的动作的索引
    action_idx = logits.detach().numpy().flatten().argmax()
    
    # 使用FeatureAgent将动作索引转换为平台可以理解的字符串格式
    response = agent.action2response(action_idx)
    return response

# 导入sys模块, 用于与标准输入输出交互, 特别是刷新输出缓冲区
import sys


# 一局比赛中可能收到的输入示例: 参见 example.log

# Python程序的入口点
if __name__ == '__main__':
    # 实例化神经网络模型, data_dir 是我们在 botzone 上存储模型参数的路径
    model = Net()
    data_dir = '/data/24.pkl'
    model.load_state_dict(torch.load(data_dir, map_location = torch.device('cpu'))) # map_location=torch.device('cpu') 确保模型加载到 CPU 上
    model.eval() 

    # Botzone平台在启动程序后, 会先发送一个初始化信号或者等待程序发出准备好的信号
    # 这里的 input() 可能是等待平台的初始指令或确认AI已启动
    # 例如, 平台可能会发送回合数或其他全局配置信息, 但在这个脚本中似乎只是一个简单的同步点
    input() # 等待平台的第一个输入, 通常是一个回合开始的信号或者简单的 "1"

    # 定义一些可能在游戏过程中需要的状态变量
    seatWind = -1 # AI的座位号/门风, 会在游戏开始时由平台指定
    agent = None    # FeatureAgent的实例, 会在收到座位号后初始化
    angang = None   # 记录AI自己暗杠的牌, 用于后续区分是自己的暗杠还是他人的杠
    zimo = False    # 标记其他玩家是否是摸牌状态, 用于判断其杠牌是否为暗杠

    # 主循环, 持续接收和处理来自平台的游戏指令
    while True:
        # 从标准输入读取平台发送的指令, 如果读取到空行, 则继续读取, 直到获得非空指令
        request = input()
        while not request.strip(): request = input()
        
        # 将指令按空格分割成列表
        t = request.split()
        
        # 根据指令的第一个词（指令类型）进行处理
        if t[0] == '0': 
            # 指令类型 '0': 游戏开始/初始化阶段
            # t[1] 是分配给 AI 的座位号/座风 (0-3, 代表东、南、西、北), t[2] 是场风 (E, S, W, N)
            # 例子: 0 0 0 
            # 第一个数是指令类型, 第二个数是AI的座位号, 第三个数是场风
            seatWind = int(t[1])

            # 初始化FeatureAgent, 传入AI的座位号
            agent = FeatureAgent(seatWind)

            # 将场风信息更新到 agent 的状态中
            agent.request2obs('Wind %s' % t[2])

            # 向平台回复 "PASS", 表示已收到并处理完毕, 等待下一步指令
            print('PASS')
        elif t[0] == '1':
            # 指令类型 '1': 发牌阶段
            # t[1] 到 t[4] 是花牌, 在我们的赛制中被弃用, t[5:] 是AI的初始13张手牌
            # 例子: 1 0 0 0 0 W8 B3 W4 T5 T3 B9 B8 B1 T1 B9 W7 T4 F2

            # 将发牌信息更新到agent的状态中
            agent.request2obs(' '.join(['Deal', *t[5:]]))

            print('PASS')
        elif t[0] == '2':
            # 指令类型 '2': 轮到AI摸牌
            # t[1] 是AI摸到的牌
            # 例子: 2 W4

            # 将摸牌信息更新到agent的状态中, 并获取当前的观测值 (这里是一个聚合特征)
            obs = agent.request2obs('Draw %s' % t[1])
            # 根据观测值, 让模型决策下一步的动作
            response = obs2response(model, obs)
            # 解析AI的决策结果
            res_parts = response.split()
            if res_parts[0] == 'Hu': # 如果决策是 "Hu" (和牌)
                print('HU')

            elif res_parts[0] == 'Play': # 如果决策是 "Play" (弃牌)
                print('PLAY %s' % res_parts[1]) # res_parts[1] 是要打出的牌
                
            elif res_parts[0] == 'Gang': # 如果决策是 "Gang" (这里特指暗杠) [玩家摸牌后, 选择杠牌, 如果成功, 则玩家会再次摸牌 (由平台指令)]
                print('GANG %s' % res_parts[1]) # res_parts[1] 是杠的牌
                angang = res_parts[1]

            elif res_parts[0] == 'BuGang': # 如果决策是 "BuGang" (补杠) [玩家摸牌摸到一张牌, 之前碰过这张牌, 进行补杠, 成功后会再次摸牌]
                print('BUGANG %s' % res_parts[1]) # res_parts[1] 是补杠的牌


        elif t[0] == '3':
            # 指令类型 '3': 其他玩家的动作
            p = int(t[1]) # 动作发生的玩家座位号

            action_type = t[2] # 动作类型 (DRAW, GANG, BUGANG, CHI, PENG, PLAY)

            if action_type == 'DRAW':
                # 其他玩家摸牌
                agent.request2obs('Player %d Draw' % p) # 记录信息
                zimo = True # 标记该玩家处于摸牌状态, 后续的杠可能是暗杠
                print('PASS')

            elif action_type == 'GANG':
                # 其他玩家杠牌
                if p == seatWind and angang:
                    # 如果是 AI 自己 (从 seatWind 看出来) 刚刚声明的暗杠（由指令'2'中的GANG触发）, agent已经处理过
                    # 此时平台给出的回复说明这个暗杠已经成功, 需要更新 agent 的状态
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                    angang = None # 重置angang标记

                elif zimo: # 如果该玩家是摸牌后杠, 则为暗杠
                    agent.request2obs('Player %d AnGang' % p)

                else: # 否则为明杠（大明杠, 杠了别人打的牌） [*]
                    agent.request2obs('Player %d Gang' % p)

                print('PASS')

            elif action_type == 'BUGANG':
                # 其他玩家补杠
                # t[3] 是补杠的牌
                # 更新agent状态, 并获取观测值, 因为AI可能有机会抢杠和
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                if p == seatWind:
                    # 如果是 AI 自己补杠, 由于刚才已经记录了信息, 这里不需要再处理
                    print('PASS')
                else:
                    # 其他玩家补杠, 由于补杠操作可能会导致 AI 有机会 ``抢杠和``, 
                    # 如果 AI 听的牌，恰好就是补杠玩家用来补杠的那一张牌，那么你就可以立即宣布“胡！”。
                    # 抢杠胡是一种比较少见但很刺激的和牌方式, 因为它打断了对方的一个有利行动（杠牌通常能带来额外的摸牌机会和潜在的番数增加）。
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')

            else: # 其他玩家执行了非摸牌、非杠牌的动作, 也就是 吃/碰/弃牌
                zimo = False # 当场上某个玩家打牌后, 此时没有有玩家处于自摸状态, 不会有暗杠的可能

                if action_type == 'CHI':
                    # 其他玩家吃牌
                    # 例子: 3 1 CHI T2 J1
                    # t[3] 是吃的牌形成的顺子中的中心牌 (如这里的 T2)
                    agent.request2obs('Player %d Chi %s' % (p, t[3]))

                elif action_type == 'PENG':
                    # 其他玩家碰牌
                    # 例子: 3 3 PENG W1
                    # 这里之所以不记录碰牌的信息, 是因为可以通过上一回的出牌信息反推
                    agent.request2obs('Player %d Peng' % p)
                
                # 无论是直接打牌, 还是吃碰后打牌, 最后都会有一张牌被弃掉, t[-1] 是被打出的牌
    
                # 更新agent状态关于其他玩家打牌的信息, 并获取观测值
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                

                # 如果进行的是弃牌! 那么我们终于有操作的空间了

                # 如果是自己弃的牌, 那么我们不需要做任何事情
                # 例如: 3 0 PLAY T1
                if p == seatWind:
                    print('PASS')
                else:
                    # 如果弃牌的是其他玩家, 那么我们需要判断是否可以和、碰、杠、吃
                    # 例如 3 3 PLAY W9

                    # 让模型决策下一步的动作
                    response = obs2response(model, obs)
                    res_parts = response.split()
                    if res_parts[0] == 'Hu': # 和牌
                        print('HU')

                    elif res_parts[0] == 'Pass': # 过
                        print('PASS')

                    elif res_parts[0] == 'Gang': # 杠这张打出的牌（大明杠）
                        # 为什么杠不需要特殊处理?
                        # 核心在于, 杠的时候不需要弃牌, 此时我们也暂时不记录自己进行了杠牌操作 (因为也不知道成不成功)
                        # 等到之后平台给出反馈时, 我们再按照之前的逻辑处理 (参见[*])
                        print('GANG') 
                        angang = None # 非暗杠

                    elif res_parts[0] in ('Peng', 'Chi'): # 碰或吃
                        
                        # 1. 假设自己已经成功碰/吃了, 由于碰/吃之后我们要立刻打出一张牌,
                        # 所以我们需要更新 agent 的状态, 让它知道我们已经执行了碰/吃操作
                        obs_for_discard = agent.request2obs('Player %d %s' % (seatWind, response)) # 更新agent状态为AI已执行碰/吃
                        
                        # 2. AI 决策碰/吃之后要打出的牌
                        response_after_action = obs2response(model, obs_for_discard)
                        
                        # 3. 向平台发送完整的动作：碰/吃 + 打出的牌
                        print(' '.join([response.split()[0].upper(), *response.split()[1:], response_after_action.split()[-1]]))
                        
                        # 4. 恢复状态, 刚才的操作我们只是假设成功了, 以实现让 agent 在对应的情况下决策
                        # 因此我们要将状态恢复到未碰/吃的状态
                        agent.request2obs('Player %d Un%s' % (seatWind, response.split()[0])) # 例如: Player 0 UnPeng

        # Botzone 平台要求在每次输出后打印这个特殊字符串, 以保持连接并表示 AI 仍在运行
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        # 刷新标准输出缓冲区, 确保信息立即发送给平台
        sys.stdout.flush()