# 数据预处理的逻辑, 用于处理监督学习数据
from feature_timeseries import FeatureAgentTimeSeries
from typing import List, Dict
import numpy as np
import json
import argparse
from rich import print
obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
matchid = -1

l = []

# TODO: 这里的会把那些只能 PASS 的数据也过滤掉, 但是按照我们目前的设计,
# 例如 UnChi 发生时, 虽然我们甚至不需要其给我们动作, 但是我们依然需要这个 obs 能够被正确输入模型, 
# 简单的想法是, 让 action 长度和 obs 一样, 但是那些不要求输出的部分, 我们不记录其损失,
# 例如标记 action 为 114514, 在下面 filter 的逻辑中, 同时判断 o['action_mask'].sum() > 1 和 actions[i] != 114514, 这样就可以了
def filterData():
    global obs
    global actions
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1: # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions

def saveData():
    x:List[Dict]
    try:
        assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    except AssertionError as e:
        import ipdb; ipdb.set_trace()
    l.append(sum([len(x) for x in obs]))
    import os
    os.makedirs('data', exist_ok=True)
    
    lengths = [[] for i in range(4)]
    
    for i in range(4):
        lengths[i] = [len(obs['event_list']) for obs in obs[i]]
        for j in range(1,len(obs[i])):
            lengths[i][j] = lengths[i][j-1] + lengths[i][j]
    assert min(lengths[i][0] for i in range(4)) == 1
    histories = []
    for i in range(4):
        try:
            histories.append(np.stack([event for x in obs[i] for event in x['event_list'] ]).astype(np.int8))
        except Exception as e:
            print(f"Error processing history for agent {i} {e}")
            import ipdb;ipdb.set_trace()

    L = len(histories[0])
    
    
    for i in range(1,4):
        try:
            assert len(histories[i]) == L, f"Length mismatch for histories[{i}]: {[len(histories[i]) for i in range(4)]} != {L}"
        except AssertionError as e:
            print(e)
            print(f'action num :{sum(len(act) for act in actions)}')
            import ipdb;ipdb.set_trace()
    
                  
    for i in range(4):        
        obs[i] = obs[i][:-1]
        actions[i] = actions[i][:-1]
        lengths[i] = lengths[i][:-1]
        np.savez(f'data/{matchid*4+i}.npz'
            , history = histories[i] 
            , mask = np.stack([x['action_mask'] for x in obs[i]]).astype(np.int8)
            , act = np.array(actions[i])
            , global_state = np.stack([x['global_state'] for x in obs[i]]).astype(np.int8)
            , lengths = np.array(lengths[i]).astype(np.int32)
        )
    for x in obs: x.clear()
    for x in actions: x.clear()

def checklength(agents:List[FeatureAgentTimeSeries],line):
    lengths = []
    for i in range(4):
        lengths.append(sum([len(obs['event_list']) for obs in obs[i]])+len(agents[i].event_pool))
    for j in range(1,4):
        if lengths[j] != lengths[j-1]:
            print(f'lengths not matching! {lengths}')
            print(line)
            import ipdb;ipdb.set_trace()
            
def event_to_string(event_vec, agentidx=0):
    """
    将85维EVENT向量转换为可读字符串，供调试使用
    
    参数:
        event_vec: 85维numpy数组, 包含玩家、动作类型和牌信息
        agentidx: 观察者的绝对位置索引(0-3)，用于将相对位置转换为绝对位置
        
    返回:
        str: 格式化的可读字符串, 例如"Player 1 PLAY_TILE W3"
    """
    # 定义常量 (与FeatureAgentTimeSeries中保持一致)
    ACTION_TYPES = [
        'DRAW_TILE', 'PLAY_TILE', 'PENG', 'CHI', 'GANG', 
        'BUGANG', 'ANGANG', 'WIN', 'NO_ACTION'
    ]
    
    TILE_LIST = [
        *[f'W{i+1}' for i in range(9)],  # 万
        *[f'T{i+1}' for i in range(9)],  # 筒
        *[f'B{i+1}' for i in range(9)],  # 饼
        *[f'F{i+1}' for i in range(4)],  # 风
        *[f'J{i+1}' for i in range(3)]   # 箭
    ]
    NONE_TILE_STR = "None"
    TILES_FOR_FEATURE = TILE_LIST + [NONE_TILE_STR]
    
    # 分解向量
    player_vec = event_vec[:4]
    action_type_vec = event_vec[4:13]
    card1_vec = event_vec[13:48]
    card2_vec = event_vec[48:83]
    
    # 获取最大值索引
    player_rel = -1 if sum(player_vec) == 0 else player_vec.argmax()
    action_type_idx = action_type_vec.argmax()
    card1_idx = card1_vec.argmax()
    card2_idx = card2_vec.argmax()
    
    # 将相对位置转换为绝对位置
    player_abs = (player_rel + agentidx) % 4 if player_rel >= 0 else -1
    
    # 获取对应的字符串表示
    player_str = "System" if player_abs == -1 else f"Player {player_abs}"
    action_type_str = ACTION_TYPES[action_type_idx]
    card1_str = TILES_FOR_FEATURE[card1_idx]
    card2_str = TILES_FOR_FEATURE[card2_idx]
    
    # 根据动作类型生成描述性输出
    if action_type_str == 'NO_ACTION':
        return f"{player_str} NO_ACTION"
    elif action_type_str == 'DRAW_TILE':
        return f"{player_str} Draw {card1_str}"
    elif action_type_str == 'PLAY_TILE':
        return f"{player_str} Play {card1_str}"
    elif action_type_str == 'CHI':
        return f"{player_str} Chi {card1_str}"
    elif action_type_str == 'PENG':
        return f"{player_str} Peng {card1_str}"
    elif action_type_str == 'GANG':
        return f"{player_str} Gang {card1_str}"
    elif action_type_str == 'BUGANG':
        return f"{player_str} Bugang {card1_str}"
    elif action_type_str == 'ANGANG':
        card_display = card1_str if card1_str != NONE_TILE_STR else "Unknown"
        return f"{player_str} Angang {card_display}"
    elif action_type_str == 'WIN':
        return f"{player_str} WIN with {card1_str}"
    else:
        return f"{player_str} {action_type_str} {card1_str} {card2_str}"
    
                
input_pth = 'data.txt'
if __name__ == '__main__':
    with open(input_pth, encoding='UTF-8') as f:
        line = f.readline()
        while line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            if t[0] == 'Match':
                agents = [FeatureAgentTimeSeries(i) for i in range(4)]
                matchid += 1
                if matchid % 1 == 0:
                    print('Processing match %d %s...' % (matchid, t[1]))
            elif t[0] == 'Wind':
                for agent in agents:
                    agent.request2obs(line)
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                            actions[p].append(0)
                        else:
                            # TODO: 即使不是自己摸牌, 也需要把事情告诉所有 model
                            # 比如要把 obs 中添加这个返回值
                            # 而 action 可能留空, 或者也可以用别的方法处理
                            agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action(' '.join(t[2:])))
                    for i in range(4):
                        if i == p:
                            # TODO: 自己弃牌也要记录下来, 不过没有 action
                            agents[p].request2obs(line)
                        else:
                            obs[i].append(agents[i].request2obs(line))
                            actions[i].append(0)
                    curTile = t[3]
                elif t[2] == 'Chi':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                            actions[p].append(0)
                        else:
                            # TODO: 其他人吃了这件事也要告诉 model, 故要写入 obs
                            agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'Peng':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                            actions[p].append(0)
                        else:
                            # TODO: 其他人碰了这件事也要告诉 model, 故要写入 obs
                            agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                elif t[2] == 'Gang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                    for i in range(4):
                        # TODO: 其他人杠了这件事也要告诉 model, 故要写入 obs
                        agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                elif t[2] == 'AnGang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                    for i in range(4):
                        # TODO: 其他人暗杠了这件事也要告诉 model, 故要写入 obs
                        if i == p:
                            agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                        else:
                            agents[i].request2obs('Player %d AnGang' % p)
                elif t[2] == 'BuGang':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            # TODO: 自己补杠了, 这件事也要告诉 model, 故要写入 obs
                            agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                        else:
                            obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                            actions[i].append(0)
                elif t[2] == 'Hu':
                    actions[p].pop()
                    actions[p].append(agents[p].response2action('Hu'))
                if t[2] !=  "Deal":
                    
                    pass #checklength(agents, line)
                # Deal with Ignore clause
                if t[2] in ['Peng', 'Gang', 'Hu']:
                    for k in range(5, 15, 5):
                        if len(t) > k:
                            p = int(t[k + 1])
                            if t[k + 2] == 'Chi':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))

                            elif t[k + 2] == 'Peng':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))

                            elif t[k + 2] == 'Gang':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                            elif t[k + 2] == 'Hu':
                                actions[p].pop()
                                actions[p].append(agents[p].response2action('Hu'))
                        else: break
            elif t[0] == 'Score':
                #filterData()
                for i in range(4):
                    # obs[i][-1]['event_list'].extend(agents[i].event_pool)
                    obs[i].append({'event_list': agents[i].event_pool})
                    actions[i].append(114514)
                saveData()
            line = f.readline()
    with open('count.json', 'w') as f:
        json.dump(l, f)