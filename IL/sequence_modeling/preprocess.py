# 数据预处理的逻辑, 用于处理监督学习数据
from feature import FeatureAgent
import numpy as np
import json
import argparse
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
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    l.append(sum([len(x) for x in obs]))
    import os
    os.makedirs('data', exist_ok=True)
    np.savez('data/%d.npz'%matchid
        , obs = np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , mask = np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , act = np.array([x for i in range(4) for x in actions[i]])
    )
    for x in obs: x.clear()
    for x in actions: x.clear()

input_pth = 'sample.txt'
with open(input_pth, encoding='UTF-8') as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            if matchid % 128 == 0:
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
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                            # TODO: 我们同时还需要让模型熟悉 UnChi 的情况
                            # 这里需要添加 obs, 对应于 UnChi, 让我们的模型学会如何从 UnChi 中恢复, action 同样特殊处理?
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                            # TODO: 我们同时还需要让模型熟悉 UnPeng 的情况
                            # 这里需要添加 obs, 对应于 UnPeng, 让我们的模型学会如何从 UnPeng 中恢复, action 同样特殊处理?
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else: break
        elif t[0] == 'Score':
            filterData()
            saveData()
        line = f.readline()
with open('count.json', 'w') as f:
    json.dump(l, f)