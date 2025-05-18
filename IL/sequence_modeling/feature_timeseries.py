from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
   # raise

# FeatureAgent 的定位是什么样的?

#######################################################################################################################
# 比赛平台 ----> __main__.py ---(request)---> FeatureAgent ---(obs)---> Net(模型) ---(action)---> FeatureAgent ---(request)---> __main__.py ----> 比赛平台#
#######################################################################################################################

# 1. FeatureAgent 负责接受从收集的信息中提取特征, 将其记忆下来, 并且整合为模型容易处理的形式
# 2. 简单来说, (按照目前的设计) 我们的模型不负责记忆任何对局信息, 只负责基于 FeatureAgent 提供的整合信息来做决策
# 3. 我们的 FeatureAgent 除了要提供 obs 以外, 还要提供 action_mask, 用来滤除不合法的动作.
# 4. FeatureAgent 可以理解为一个从原始信息到 obs 的映射, 不过这个映射是带有记忆的.
# 5. FeatureAgent 还要负责解析动作, 将模型的输出转换为平台可以理解的动作, 不过我们基本不会改变这一部分的设计

class FeatureAgentTimeSeries(MahjongGBAgent):

    # 每一次获取的事件为一个大小为 83 的向量.
    EVENT_SIZE = 83

    GLOBAL_CHANNAL = 14

    GLOBAL_OFFSET = {
        'SEAT_WIND' : 0,        # 座风      通道 0
        'PREVALENT_WIND' : 1,   # 场风      通道 1
        'HAND' : 2,             # 己方手牌  通道 2-5 (共 4 个)
        'SHOW' : 6,             # 已经打出的牌 通道 6-9 (共 4 个)
        'REMAIN0' : 10,        # 剩余牌数
        'REMAIN1' : 11,        # 剩余牌数
        'REMAIN2' : 12,        # 剩余牌数
        'REMAIN3' : 13         # 剩余牌数
    }

    # 我们还需要提供一个 action_mask, 由于设计的动作维度是 235, 所以 action_mask 的大小是 235, 不过我们应该不会改变这一部分
    ACT_SIZE = 235
    

        # ACT 的偏移，我觉得我们不会调整它
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }

    # 牌名列表, 我们解析信息时需要用到它
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)), # 万
        *('T%d'%(i+1) for i in range(9)), # 筒
        *('B%d'%(i+1) for i in range(9)), # 饼
        *('F%d'%(i+1) for i in range(4)), # 风
        *('J%d'%(i+1) for i in range(3))  # 箭
    ]
    # 牌名 -> 索引 的一个字典，这个索引的范围是 OBS 的第二个维度 36（实际上最后有空余）
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}

    # 扩展原先的 OFFSET_ACT, 使其包含原先的 NONE_TILE
    NONE_TILE_STR = "None"
    TILES_FOR_FEATURE = TILE_LIST + [NONE_TILE_STR]
    TILE_TO_IDX_FEATURE = {tile: i for i, tile in enumerate(TILES_FOR_FEATURE)}
    NONE_TILE_IDX_FEATURE = TILE_TO_IDX_FEATURE[NONE_TILE_STR]
    # print(TILE_TO_IDX_FEATURE)

    # ===== 定义 event 向量 =====

    # 1. 执行动作的玩家: 4 维 one-hot 向量
    
    # 2. 动作类型向量部分: 9 维 one-hot 向量
    ACTION_TYPES_DEF = {
        'DRAW_TILE': 0,
        'PLAY_TILE': 1,
        'PENG': 2,
        'CHI': 3,
        'GANG': 4,
        'BUGANG': 5,
        'ANGANG': 6,
        'WIN': 7,
        'NO_ACTION': 8
    }
    NUM_ACTION_TYPES = 9

    # 3. 可能涉及的第一张牌 (Primary card): 35 维 one-hot 向量, 其中一个涉及的是 None

    # 4. 可能涉及的第二张牌 (Secondary card): 35 维 one-hot 向量, 其中一个涉及的是 None
    

    # 定义初始化函数, 这里面定义的变量可以理解为是记忆存储的地方
    def __init__(self, seatWind):        
        # 座风, 玩家的座位, 由于 座风信息 与上面的风牌有一定关联 (在番数上), 所以我们记录在 obs['SEAT_WIND'] 的风牌位置处 (0/1)
        # 例如, 假设是东风, 那么在 obs['SEAT_WIND'] 这一通道的 `东风` (F1?) 的位置处会置为 1, 其他的风牌通道会置为 0

        self.seatWind = seatWind

        # 记录每个玩家的吃、碰、杠牌的组合, 初始时都为空
        self.packs = [[] for i in range(4)]

        # 记录每个玩家的出牌历史, 初始时都为空
        self.history = [[] for i in range(4)]

        # 记录每个玩家剩余的牌墙数, 用来把控比赛的进程
        self.tileWall = [21] * 4

        # 记录已经展示出来的牌的数量, 目前为空, 未来将是一个从 牌名 到 数量 的字典
        # 其包括弃牌以及吃碰杠的牌 (展示出来的)
        self.shownTiles = defaultdict(int)

        # 用于临时标识 下家 牌墙还有没有牌剩余
        # 这是为了处理在国标麻将中有特殊的情况 [妙手回春/海底捞月]
        # 这样的情况会对算番造成影响, 因此我们必须要考虑!
        self.wallLast = False

        # 标识是否刚刚进行了杠牌, 这里特别处理 [岭上开花], 也就是杠了牌之后摸牌引起的胡牌
        self.isAboutKong = False
        
        # 这些感觉都可以放在后面初始化, python 不要求预先定义成员变量

        # Variables for constructing the timeseries embedding
        self.current_event_player_relative = -1 # -1 for system/no specific player, 0-3 for players relative to self
        self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
        self.current_event_card1_str = self.NONE_TILE_STR
        self.current_event_card2_str = self.NONE_TILE_STR
        
        self.prevalentWind = -1

        self.hand = [] 
        self.valid = []
        self.curTile = None
        self.tileFrom = -1
        self.event_pool = []

        self.global_obs = np.zeros((self.GLOBAL_CHANNAL, 36), dtype=np.float32)

        # 更新座风信息
        self.global_obs[self.GLOBAL_OFFSET['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1 #* 在风牌的位置放一个1


    def _reset_current_event_info(self):
        """Resets event-specific information at the start of each request processing."""
        self.current_event_player_relative = -1 
        self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
        self.current_event_card1_str = self.NONE_TILE_STR
        self.current_event_card2_str = self.NONE_TILE_STR
        self.valid = [] # Clear valid actions for the new state

    def request2obs(self, request: str):
        """
        根据一行请求更新观测信息 self.obs。
        请求的类型可以多样化，下述任意一行均为有效：
            Wind 0..3
            Deal XX XX ...
            Player N Draw
            Player N Gang
            Player N(me) AnGang XX
            Player N(me) Play XX
            Player N(me) BuGang XX
            Player N(not me) Peng
            Player N(not me) Chi XX
            Player N(not me) AnGang
            
            Player N Hu
            Huang
            Player N Invalid
            Draw XX
            Player N(not me) Play XX
            Player N(not me) BuGang XX
            Player N(me) Peng
            Player N(me) Chi XX
        """

        self._reset_current_event_info()
        
        t = request.split()
        action_type_str = t[0]

        if action_type_str == 'Wind':
            """
            Wind [0~3]
            """
            # 存储当前场风
            self.prevalentWind = int(t[1])
            # 更新场风信息
            self.global_obs[self.GLOBAL_OFFSET['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            
            return

        elif action_type_str == 'Deal':

            self.hand = list(t[1:]) 
            
            return
        
        elif action_type_str == 'Huang':
            self.current_event_player_relative = -1
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION'] 
            self.valid = []
            return self._obs()

        elif action_type_str == 'Draw':

            self.tileWall[self.seatWind] -= 1
            
            # 检查下家是否是最后一张牌
            self.wallLast = self.tileWall[1] == 0
            
            # 获取摸到的牌
            tile = t[1]

            self.current_event_player_relative = 0
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['DRAW_TILE']
            self.current_event_card1_str = tile
            self.append_event()

            # 初始置空可用操作空间
            self.valid = []

            # 检查是否能胡牌 (通过封装的算番函数)
            
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            

            # 重置杠相关标记, 具体来说, 我们的上一次操作不再是杠牌, 也就消除了 [岭上开花] 的情况
            self.isAboutKong = False

            self.hand.append(tile)

            # 生成动作空间
            for tile in set(self.hand): # set 可以去重
                # 生成打出牌的有效动作，添加 Play 牌X 的动作索引
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                # 为什么这里后面的逻辑要检查 self.wallLast 呢? 我的理解: 这是因为如果其是 True, 那么自己此时的一手会是最后一个弃牌的, 如果前面 _check_mahjong 表明自己胡不了了, 那么自己必然胡不了了 (不管是暗杠还是什么都一样), 于是就不考虑这些操作了.


                # 检查暗杠的可能性，添加 AnGang 牌X 的动作索引
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
                    
            # 检查补杠的可能性
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])

            # 因为摸牌后我们一定需要执行某个操作, 所以我们需要返回一个 obs
            return self._obs()

        # 往后的分支都是根据 Player N XX 这种格式的请求来处理的
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        # 他们是从上帝视角描述的, 但不等于都不是自己做的
        # 获取玩家编号 (0 是自己)

        p_abs = int(t[1])
        p_rel = (p_abs + 4 - self.seatWind) % 4
        event_str = t[2]
        
        self.current_event_player_relative = p_rel

        # Player N Draw
        if event_str == 'Draw': # Other player draws
            """
            玩家 p 摸牌
            """
            # 减少玩家 p 的牌墙数量
            self.tileWall[p_rel] -= 1

            # 检查下家是否是最后一张牌, 因为这个时候当前玩家要弃牌, 如果下家牌堆没有牌了, 那么就可能触发 [妙手回春/海底捞月]
            self.wallLast = self.tileWall[(p_rel + 1) % 4] == 0

            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['DRAW_TILE']


            # TODO: 这里把其他人摸了牌的事件存入 event_list
            self.append_event()
            return

        # Player N Invalid
        if event_str == 'Invalid': 
            """
            玩家 p 的操作无效
            """
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
            # 初始置空可用操作空间
            self.valid = []

            # TODO: 这里把其他人无效的事件存入 event_list
            self.append_event()
            return self._obs()
        
        # Player N Hu
        if event_str == 'Hu':
            """
            玩家 p 胡牌
            """
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['WIN']
            # Winning tile is typically self.curTile (set by Draw, Play, or Gang/BuGang that was Hu'd)
            self.current_event_card1_str = self.curTile if self.curTile else self.NONE_TILE_STR

            # 初始置空可用操作空间
            self.valid = []

            # TODO: 这里把其他人胡牌的事件存入 event_list
           # self.append_event() #! Xia modify
            return self._obs()
        
        # Player N Play XX [其他玩家弃牌了]
        if event_str == 'Play':
            """
            玩家 p 打出一张牌 XX
            """
            # 获取打出的牌
            self.tileFrom = p_rel
            self.curTile = t[3]

            # 更新打出的牌数量 (这是一个全局的字典, 例如目前场上一共打出了多少张 五万)
            self.shownTiles[self.curTile] += 1 
            self.history[p_rel].append(self.curTile)


            # 设置了信息
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['PLAY_TILE']
            self.current_event_card1_str = self.curTile

            # TODO: 这里把其他人打出的牌的事件存入 event_list
            self.append_event()

            # 如果是自己打出的牌, 那么什么也不用做
            if p_rel == 0:
                # 从手牌中移除打出的牌，更新手牌
                if self.curTile in self.hand: self.hand.remove(self.curTile)
                # self.append_event()
                return 
            
            # 如果是别人打出了某张牌, 那么就需要我们考虑 吃/碰/杠/胡/过 等操作
            else:
                # 可选动作：Hu, Gang, Peng, Chi, Pass
                self.valid = []

                # 检查是否能胡牌

                # 缺省这两个 isSelfDrawn=False, isAboutKong=False 是默认值也没事
                if self._check_mahjong(self.curTile, isSelfDrawn=False, isAboutKong=False):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                

                # 检查是否是最后一张牌, 如果是最后一张牌, 意味着不成功便成仁, 这张弃牌要么能让自己胡牌 (刚才考虑了), 要么胡不了, 就只考虑 PASS

                # 如果不是最后一张牌, 那么我们还有一些余地, 考虑其他的操作
                if not self.wallLast:

                    # 检查碰和杠的可能性

                    # 如果有两张相同的牌, 则可以碰牌
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])

                        # 如果自己有 3 张相同的牌, 并且自己的牌墙还有牌, 则可以杠牌 (因为杠了之后要补牌, 必须要从自己牌墙拿)
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])

                    # 检查吃的可能性
                    color = self.curTile[0]
                    if p_rel == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)

                # 添加 Pass 动作
                self.valid.append(self.OFFSET_ACT['Pass'])

                return self._obs()
            
        # Player N Chi XX
        if event_str == 'Chi':
            """
            玩家 p 吃牌 XX
            """
            # 这是顺子的 **中心牌**, 未必是被吃的牌
            # middle_tile_chi is string
            middle_tile_chi = t[3] 
            color = middle_tile_chi[0]
            # 这是中心牌的序号 (例如顺子 W4, W5, W6 中的 W5)
            num = int(middle_tile_chi[1])
            eaten_tile = self.curTile

            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['CHI']
            self.current_event_card1_str = middle_tile_chi
            self.current_event_card2_str = eaten_tile # Eaten tile is secondary info for Chi event

            # TODO: 这里把其他人吃牌的事件存入 event_list
            self.append_event()

            # 这里的 middle_tile_chi[1] 存储了被吃的牌的序号 (例如被吃的是 W6, 形成顺子 W4, W5, W6)
            # 这里记录的第一个是 行为, 第二个是 中心牌, 第三个是被吃的牌在顺子中的位置, i = 1, 2, 3
            self.packs[p_rel].append(('CHI', num, int(middle_tile_chi[1]) - num + 2))

            # 之前我们在弃牌的时候将 self.shownTiles[self.curTile] + 1, 意思是弃牌堆中某种牌的数量 + 1
            # 但是现在它落到了玩家 p 的手牌中, 变成了不可见的, 所以我们要将它 -1
            self.shownTiles[self.curTile] -= 1

            # 形成的顺子会被展示出来
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1

            # 检查下家是否是最后一张牌, 因为吃完后要弃牌, 上家弃牌可能会触发 [妙手回春/海底捞月]
            self.wallLast = self.tileWall[(p_rel + 1) % 4] == 0

            # 如果收到的 request 是自己吃了牌 [注意这里只是假设自己吃成功了!]
            # 那么我们要考虑弃什么牌
            if p_rel == 0:
                # 首先我们假装吃成功了, 然后我们要从手牌中移除被吃的牌
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))

                # 考虑弃牌, 手上的牌都可以弃
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])

                return self._obs()

            else:
                return 
            
        # Player N UnChi XX
        # 这仅仅发生在自己先假设自己吃了牌, 但是最终平台告诉你吃牌失败了 (因为杠/碰/胡优先级更高)
        if event_str == 'UnChi':
            """
            玩家 p 取消吃牌 XX
            """

            # 获取取消吃的牌
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            # 更新取消吃牌的记录
            self.packs[p_rel].pop()
            self.shownTiles[self.curTile] += 1

            # 总之需要还原, 具体不用管
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p_rel == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)

            return 

        # Player N Peng
        if event_str == 'Peng':
            """
            玩家 p 碰牌
            """
            peng_tile = self.curTile # Tile being Peng-ed (set by previous Play event)
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['PENG']
            self.current_event_card1_str = peng_tile

            # TODO: 这里把其他人碰牌的事件存入 event_list
            self.append_event()

            # 更新碰牌的记录
            # 这里第一个是 行为, 第二个是 碰牌, 第三个则是相对于碰牌者的相对位置, 上家是 1, 对家是 2, 下家是 3
            self.packs[p_rel].append(('PENG', self.curTile, (4 + p_rel - self.tileFrom) % 4))

            self.shownTiles[self.curTile] += 2

            # 检查是否是最后一张牌, 类似地, 考虑下家是否是最后一张牌
            self.wallLast = self.tileWall[(p_rel + 1) % 4] == 0

            # 同样还是假设自己碰成功了, 然后我们要从手牌中移除被吃的牌
            if p_rel == 0:
                # 可选动作：Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)

                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                
                return self._obs()
            else:
                return 

        # Player N UnPeng
        if event_str == 'UnPeng':
            """
            玩家 p 取消碰牌
            """

            # 更新取消碰牌的记录
            self.packs[p_rel].pop()
            self.shownTiles[self.curTile] -= 2
            if p_rel == 0:
                for i in range(2):
                    self.hand.append(self.curTile)

            return 
        
        if event_str == 'Gang':
            """
            玩家 p 杠牌
            """
            # Modify:
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['GANG']
            self.current_event_card1_str = self.curTile

            # TODO: 这里把其他人杠牌的事件存入 event_list
            self.append_event()

            # 更新杠牌的记录
            # 和碰差不多, 上家是 1, 对家是 2, 下家是 3
            self.packs[p_rel].append(('GANG', self.curTile, (4 + p_rel - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3

            # 假设自己杠成功了, 然后我们要从手牌中移除被吃的牌
            if p_rel == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self.isAboutKong = True

            return 
        
        # Player N AnGang XX
        if event_str == "AnGang":
            """
            玩家 p 暗杠 XX
            """
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['ANGANG']

            # TODO: 这里把其他人暗杠的事件存入 event_list
            self.append_event()

            # 如果是自己暗杠, 才知道是什么牌
            # 如果其他人暗杠了, 那么我们只知道是暗杠, 但是不知道是什么牌
            tile = 'CONCEALED' if p_rel else t[3]

            # 更新暗杠的记录
            self.packs[p_rel].append(('GANG', tile, 0))

            if p_rel == 0:
                self.isAboutKong = True
                self.current_event_card1_str = tile

                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
                self.current_event_card1_str = self.NONE_TILE_STR

            return 

        # Player N BuGang XX
        if event_str == 'BuGang':
            """
            玩家 p 补杠 XX
            """
            # 更新补杠的记录
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['BUGANG']
            self.current_event_card1_str = self.curTile

            # TODO: 这里把其他人补杠的事件存入 event_list
            self.append_event()

            tile = t[3]
            for i in range(len(self.packs[p_rel])):
                # 在三元组中找到对应的牌, 这里的 tile 是补杠的牌
                # 因为既然能够补杠, 说明其一定是碰过的牌, 已经有 3 张了, 而还能杠, 说明不存在位于顺子的可能, 故可以这样处理
                if tile == self.packs[p_rel][i][1]:
                    # 其他不变, 仅仅是把 'PENG' 改为 'GANG'
                    self.packs[p_rel][i] = ('GANG', tile, self.packs[p_rel][i][2])
                    break
            self.shownTiles[tile] += 1
            

            if p_rel == 0:
                self.hand.remove(tile)
                self.isAboutKong = True
                return 

            else:
                # 考虑补杠胡
                # 可选动作：Hu, Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])

                return self._obs()

        # 其余的情况都是非法的
        raise NotImplementedError('Unknown request %s!' % request)


    # 一个解析模型输出的动作的函数, 这个函数会将模型输出的动作转换为平台可以理解的动作, 我们不用管他
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        """
        将动作索引转换为对应的动作字符串。
        """
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''

    # 同上, 我们不管他
    def response2action(self, response):
        """
        将动作字符串转换为对应的动作索引。
        """
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _hand_embedding_update(self):
        """
        根据 self.hand 更新 self.obs 矩阵中的手牌信息部分，以反映当前玩家手中的牌。
        前接 deal
        """
        # 清空手牌之后到暗杠数量之前的所有信息
        self.global_obs[self.GLOBAL_OFFSET['HAND']::] = 0

        # 统计手牌中各种牌的数量
        d = defaultdict(int) # d[tile] 表示手牌中牌 tile 的数量，默认值为 0
        for tile in self.hand:
            d[tile] += 1

        for tile in d:
            # 更新 self.obs 矩阵中的手牌信息部分
            # GLOBAL_OFFSET['HAND'] 是常量偏移 2，用于锁定手牌信息部分的起始位置
            # 等价于 self.obs[2:2+d[tile]]
            # 所以这里可以看出来，后面四个维度实际记录的是你拥有几张牌的信息，拥有 3 张牌则 self.obs[2:5] 都是 1
            # 也就是拥有几张就顺序记录几个维度 
            self.global_obs[self.GLOBAL_OFFSET['HAND'] : self.GLOBAL_OFFSET['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1

    def _shown_tiles_embedding_update(self):
        d = defaultdict(int) # 表示展示出来的牌的数量，默认值为 0
        for tile in self.shownTiles:
            d[tile] += 1

        for tile in d:
            self.global_obs[self.GLOBAL_OFFSET['SHOW'] : self.GLOBAL_OFFSET['SHOW'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _remains_embedding_update(self):
        for p in range(4):
            # 整个通道设置为剩余牌数
            self.global_obs[self.GLOBAL_OFFSET['REMAIN%d' % p]] = self.tileWall[p] 

    def append_event(self):
        # 更新事件信息
        # 1. Acting Player (4 bits one-hot, or all zeros if system/no specific player)
        player_vec = np.zeros(4)
        if 0 <= self.current_event_player_relative < 4:
            player_vec[self.current_event_player_relative] = 1

        # 2. Action Type (NUM_ACTION_TYPES bits one-hot)
        action_type_vec = np.zeros(self.NUM_ACTION_TYPES)
        if 0 <= self.current_event_action_type_idx < self.NUM_ACTION_TYPES: # Boundary check
            action_type_vec[self.current_event_action_type_idx] = 1
        else: # Should not happen if current_event_action_type_idx is always valid
            action_type_vec[self.ACTION_TYPES_DEF['NO_ACTION']] = 1 
        
        # 3. Primary Card (35 bits one-hot: 34 tiles + "None")
        card1_vec = np.zeros(len(self.TILES_FOR_FEATURE))
        card1_idx = self.TILE_TO_IDX_FEATURE.get(self.current_event_card1_str, self.NONE_TILE_IDX_FEATURE)
        card1_vec[card1_idx] = 1

        # 4. Secondary Card (35 bits one-hot: 34 tiles + "None")
        card2_vec = np.zeros(len(self.TILES_FOR_FEATURE))
        card2_idx = self.TILE_TO_IDX_FEATURE.get(self.current_event_card2_str, self.NONE_TILE_IDX_FEATURE)
        card2_vec[card2_idx] = 1

        event_vec = np.concatenate([
            player_vec, action_type_vec, card1_vec, card2_vec
        ])

        # 5. Event Pool (NUM_EVENT_POOL bits one-hot)
        self.event_pool.append(event_vec)

    def _obs(self):
        """
        生成当前观测信息，以及可执行动作的掩码
        """

        # 更新全局信息
        self._hand_embedding_update()
        self._shown_tiles_embedding_update()
        self._remains_embedding_update()

        # 处理需要返回的事件信息, 并且清空事件池
        event_list = self.event_pool.copy()
        self.event_pool = []

        # Action Mask
        mask = np.zeros(self.ACT_SIZE)
        # self.valid should contain valid action *indices* based on OFFSET_ACT
        if hasattr(self, 'valid') and self.valid: 
            for a_idx in self.valid:
                if 0 <= a_idx < self.ACT_SIZE: # Ensure index is within bounds
                    mask[a_idx] = 1
        
        return {
            'event_list': event_list,
            'global_state': self.global_obs.reshape((self.GLOBAL_CHANNAL, 4, 9)).copy(),
            'action_mask': mask
        }
    
    def pop_event(self):
        assert len(self.event_pool) > 0, "Event pool is empty!"
        self.event_pool.pop()
    

    # 算番函数的封装, 用来判断是否可以和牌, 或许可以借用这个接口来算番数, 作为某种间接奖励, 不过恐怕是等到基础的 RL 都完成后再精益求精才会考虑?
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        """
        检查是否可以和牌，即是否有足够的番数。
        """
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = self.shownTiles[winTile] == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True