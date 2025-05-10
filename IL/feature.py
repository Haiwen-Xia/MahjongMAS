"""
FeatureAgent继承自Agent类，按照麻将规则处理出每个决策点的所有可行动作以及简单的特征表示。
示例的特征为6*4*9，仅包含圈风、门风和自己手牌。

提示：需要修改这个类，从而支持：
1. 更加完整的特征提取（尽量囊括所有可见信息）。
"""

from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    
    '''
        
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
        235 = 过1 + 胡1 + 弃牌34 + 明杠34 + 暗杠34 + 补杠34 + 碰牌34 + 吃牌63
        其中吃牌 63 = 花色万条饼3 * 中心牌二到八7 * 吃三张中的第几张3
    '''
    # 按照README中的格式定义观察空间大小
    # 6(基本信息) + HISTORY_SIZE(历史信息)
    HISTORY_SIZE = 95  # 历史特征的维度
    BASIC_SIZE = 5  # 基本特征的维度（门风、场风、手牌）
    OBS_SIZE = BASIC_SIZE + HISTORY_SIZE  # 总观察空间大小
    
    # 定义观察空间各部分的偏移量
    OFFSET_OBS = {
        'PREVALENT_WIND': 0,  # 场风
        'SEAT_WIND': 1,       # 门风,4D one-hot
        'HISTORY': BASIC_SIZE  # 历史信息开始位置
    }
    
    # 历史特征各部分的偏移量
    OFFSET_HISTORY = {
        'ACTING_PLAYER': 0,       # 执行动作的玩家 (4维)
        'ACTION_TYPE': 4,         # 动作类型 (11维)
        'PRIMARY_CARD': 15,       # 主要涉及的牌张 (35维)
        'SECONDARY_CARD': 50,     # 次要涉及的牌张 (35维)
        'ROUND_WIND': 85,         # 圈风 (4维)
        'PLAYER_WIND': 89,        # 门风 (4维)
        'ACTION_SUCCESS': 93      # 动作成功性 (2维)
    }
    
    # 动作类型的编码
    ACTION_TYPE = {
        'DRAW': 0,               # 摸牌
        'PLAY_TILE': 1,          # 出牌
        'PENG': 2,               # 碰牌
        'CHI': 3,                # 吃牌
        'GANG': 4,               # 杠牌
        'ANGANG': 5,             # 暗杠
        'MINGGANG_DISCARD': 6,   # 明杠
        'BUGANG': 7,             # 补杠
        'PASS': 8,               # 过
        'WIN': 9,                # 和牌
        'NO_ACTION': 10          # 无动作
    }
    
    ACT_SIZE = 235

    # ACT 的偏移，也是写死的，不会变动
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
    
    # 定义初始化函数, 这里面定义的变量可以理解为是记忆存储的地方
    def __init__(self, seatWind):

        # 座风, 玩家的座位, 由于 座风信息 与上面的风牌有一定关联 (在番数上), 所以我们记录在 obs['SEAT_WIND'] 的风牌位置处 (0/1)
        # 例如, 假设是东风, 那么在 obs['SEAT_WIND'] 这一通道的 `东风` (F1?) 的位置处会置为 1, 其他的风牌通道会置为 0
        self.seatWind = seatWind

        # 记录每个玩家的出牌历史, 初始时都为空
        self.history = []

        # 记录每个玩家剩余的牌墙数, 用来把控比赛的进程
        self.tileWall = [21] * 4

        # 记录已经展示出来的牌的数量, 目前为空, 未来将是一个从 牌名 到 数量 的字典
        # 其包括弃牌以及吃碰杠的牌 (展示出来的)
        self.shownTiles = defaultdict(int)

        # 用于临时标识 下家 牌墙还有没有牌剩余
        # 这是为了处理在国标麻将中有特殊的情况 [妙手回春/海底捞月] (好像没怎么见到过)
        # 这样的情况会对算番造成影响, 因此我们必须要考虑!
        self.wallLast = False

        # 标识是否刚刚进行了杠牌, 这里特别处理 [岭上开花], 也就是杠了牌之后摸牌引起的胡牌
        self.isAboutKong = False
    
        # 标识是否暗杠 
        # TODO: 目前这里的设计并不明智!!!
        # TODO: 之后可以考虑作为一个 玩家 -> 暗杠次数 的字典, 这样可以更大程度记录相关信息
        self.AnGang = False
        
        self.AnGangTile = ''
        
        # 记录当前玩家的手牌与观测信息
        self.obs = np.zeros((len(self.history),self.OBS_SIZE, 36))
        # 更新观测信息中的座风信息
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        
        # 初始化历史特征
        self.history_features = np.zeros((self.HISTORY_SIZE))
        
        # 记录上一个动作的玩家、类型和涉及的牌
        self.last_player = -1
        self.last_action_type = -1
        self.last_primary_card = ''
        self.last_secondary_card = ''
        self.last_action_success = True
    
    def request2obs(self, request):
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
        t = request.split()

        ### 目前这里的分支都是从玩家自身的视角出发的, 都是玩家自己接收到的消息, 例如自己摸到了什么牌等到

        # 比赛开始, 我们会收到一个座风信息
        if t[0] == 'Wind':
            """
            Wind [0~3]
            """
            # 存储当前场风
            self.prevalentWind = int(t[1])
            # 更新观测信息中的场风信息
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            
            # 更新历史特征中的圈风信息
            self._update_round_wind(self.prevalentWind)
            return
        # 发牌 / 起始摸牌
        if t[0] == 'Deal':
            """
            Deal 牌1 牌2 ...
            """
            self.hand = t[1:]
            # 下面这个函数会根据 self.hand 更新 self.obs
            self._hand_embedding_update()
            return
        # 荒庄
        if t[0] == 'Huang':
            """
            Huang
            """
            # 无可用操作
            self.valid = []
            # 返回信息
            return self._obs()
        # 摸牌
        '''
        DRAW(摸牌，对应编号0)
        PLAY_TILE (出牌，对应编号1)
        PENG (碰牌并出牌，对应编号2)
        CHI (吃牌并出牌，对应编号3)
        GANG (杠牌，需要区分暗杠、明杠，对应编号4)
        ANGANG (暗杠 - 若上一回合为摸牌，对应编号5)
        BUGANG (补杠，对应编号7)
        PASS (过，例如别人打牌后，你选择不碰/杠/吃/胡，对应编号8)
        WIN (和牌，可能需要区分自摸、点炮，对应编号9)
        NO_ACTION (填充或无意义时间步，对应编号10)
        '''
        if t[0] in ['Draw','Play','Hu','Invalid','Chi','Peng','Gang','AnGang','BuGang']:
            self.history.append(' '.join(t))
        if t[0] == 'Draw':
            """
            Draw 牌1
            """
            # 可选动作： Hu, Play, AnGang, BuGang
            
            # 减少牌墙数量
            self.tileWall[0] -= 1
            
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[1] == 0
            # 获取摸到的牌
            tile = t[1]
            # 初始置空可用操作空间
            self.valid = []
            
            # 更新历史特征 - 自己摸牌
            self._update_history_feature(0, self.ACTION_TYPE['DRAW'], tile, '')
            
            # 检查是否能胡牌
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
                
            # 重置杠相关标志
            self.isAboutKong = False
            # 将摸到的牌添加到玩家手牌中，更新手牌
            self.hand.append(tile)
            self._hand_embedding_update()
            
            # 生成动作空间
            for tile in set(self.hand): # set 可以去重
                # 生成打出牌的有效动作，添加 Play 牌X 的动作索引
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                # 检查暗杠的可能性，添加 AnGang 牌X 的动作索引
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
                    
            # 检查补杠的可能性
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()#!
        
        # 往后的分支都是根据 Player N XX 这种格式的请求来处理的
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        # 获取玩家编号
        p = (int(t[1]) + 4 - self.seatWind) % 4

        # Player N Draw
        if t[2] == 'Draw':
            """
            玩家 p 摸牌
            """
            # 减少玩家 p 的牌墙数量
            self.tileWall[p] -= 1
            
            # 更新历史特征 - 其他玩家摸牌
            self._update_history_feature(p, self.ACTION_TYPE['DRAW'], '', '')
            
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return

        # Player N Invalid
        if t[2] == 'Invalid':
            """
            玩家 p 的操作无效
            """
            # 初始置空可用操作空间
            self.valid = []
            
            # 更新历史特征 - 操作无效
            self._update_action_success(False)
            
            return self._obs()

        # Player N Hu
        if t[2] == 'Hu':
            """
            玩家 p 胡牌
            """
            # 初始置空可用操作空间
            self.valid = []
            
            # 更新历史特征 - 玩家胡牌
            self._update_history_feature(p, self.ACTION_TYPE['WIN'], self.curTile if hasattr(self, 'curTile') else '', '')
            
            return self._obs()

        # Player N Play XX
        if t[2] == 'Play':
            """
            玩家 p 打出一张牌 XX
            """
            # 获取打出的牌
            self.tileFrom = p
            self.curTile = t[3]
            
            self.shownTiles[self.curTile] += 1
            # 将打出的牌记录在历史记录中
            self.history[p].append(self.curTile)
            
            # 更新历史特征 - 玩家打牌
            self._update_history_feature(p, self.ACTION_TYPE['PLAY_TILE'], self.curTile, '')
            
            # 如果是自己打出的牌
            if p == 0:
                # 从手牌中移除打出的牌，更新手牌
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # 可选动作：Hu, Gang, Peng, Chi, Pass
                self.valid = []
                # 检查是否能胡牌
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                # 检查是否是最后一张牌
                if not self.wallLast:
                    # 检查碰和杠的可能性
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    # 检查吃的可能性
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
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
        if t[2] == 'Chi':
            """
            玩家 p 吃牌 XX
            """
            # 获取吃的牌
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
                
            # 更新历史特征 - 玩家吃牌
            self._update_history_feature(p, self.ACTION_TYPE['CHI'], self.curTile, tile)
                
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        # Player N UnChi XX
        if t[2] == 'UnChi':
            """
            玩家 p 取消吃牌 XX
            """
            # 获取取消吃的牌
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            # 更新取消吃牌的记录
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
                
            # 更新历史特征 - 取消吃牌，标记为失败
            self._update_action_success(False)
                
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return

        # Player N Peng
        if t[2] == 'Peng':
            """
            玩家 p 碰牌
            """
            # 更新碰牌的记录
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            
            # 更新历史特征 - 玩家碰牌
            self._update_history_feature(p, self.ACTION_TYPE['PENG'], self.curTile, '')
            
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        # Player N UnPeng
        if t[2] == 'UnPeng':
            """
            玩家 p 取消碰牌
            """
            # 更新取消碰牌的记录
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            
            # 更新历史特征 - 取消碰牌，标记为失败
            self._update_action_success(False)
            
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return

        # Player N Gang
        if t[2] == 'Gang':
            """
            玩家 p 杠牌
            """
            # 更新杠牌的记录
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            
            # 更新历史特征 - 玩家明杠
            self._update_history_feature(p, self.ACTION_TYPE['MINGGANG_DISCARD'], self.curTile, '')
            
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return

        # Player N AnGang XX
        if t[2] == 'AnGang':
            """
            玩家 p 暗杠 XX
            """
            # 更新暗杠的记录
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            
            # 更新历史特征 - 玩家暗杠
            if p == 0:
                self._update_history_feature(p, self.ACTION_TYPE['ANGANG'], t[3], '')
            else:
                self._update_history_feature(p, self.ACTION_TYPE['ANGANG'], '', '')
                
            if p == 0:
                self.isAboutKong = True
                self.AnGang = True
                self.AnGangTile = tile
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return

        # Player N BuGang XX
        if t[2] == 'BuGang':
            """
            玩家 p 补杠 XX
            """
            # 更新补杠的记录
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            
            # 更新历史特征 - 玩家补杠
            self._update_history_feature(p, self.ACTION_TYPE['BUGANG'], tile, '')
            
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # 可选动作：Hu, Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    def _update_history_feature(self, player, action_type, primary_card, secondary_card):
        """
        更新历史特征
        """
        # 记录当前动作信息
        self.last_player = player
        self.last_action_type = action_type
        self.last_primary_card = primary_card
        self.last_secondary_card = secondary_card
        self.last_action_success = True
        
        # 重置历史特征
        self.history_features = np.zeros(self.HISTORY_SIZE)
        
        # 设置执行动作的玩家
        if player >= 0:
            self.history_features[self.OFFSET_HISTORY['ACTING_PLAYER'] + player] = 1
        
        # 设置动作类型
        if action_type >= 0:
            self.history_features[self.OFFSET_HISTORY['ACTION_TYPE'] + action_type] = 1
        
        # 设置主要涉及的牌张
        if primary_card and primary_card in self.OFFSET_TILE:
            self.history_features[self.OFFSET_HISTORY['PRIMARY_CARD'] + self.OFFSET_TILE[primary_card]] = 1
        
        # 设置次要涉及的牌张
        if secondary_card and secondary_card in self.OFFSET_TILE:
            self.history_features[self.OFFSET_HISTORY['SECONDARY_CARD'] + self.OFFSET_TILE[secondary_card]] = 1
        
        # 设置圈风
        if hasattr(self, 'prevalentWind'):
            self.history_features[self.OFFSET_HISTORY['ROUND_WIND'] + self.prevalentWind] = 1
        
        # 设置门风
        self.history_features[self.OFFSET_HISTORY['PLAYER_WIND'] + self.seatWind] = 1
        
        # 设置动作成功性
        self.history_features[self.OFFSET_HISTORY['ACTION_SUCCESS']] = 1
        
        # 将历史特征复制到观察状态的历史部分
        for i in range(self.HISTORY_SIZE):
            if self.history_features[i] > 0:
                self.obs[self.OFFSET_OBS['HISTORY'] + i // 36, i % 36] = 1
    
    def _update_action_success(self, success):
        """
        更新动作成功性
        """
        self.last_action_success = success
        
        # 清除之前的成功性标记
        self.history_features[self.OFFSET_HISTORY['ACTION_SUCCESS']] = 0
        self.history_features[self.OFFSET_HISTORY['ACTION_SUCCESS'] + 1] = 0
        
        # 设置新的成功性标记
        if success:
            self.history_features[self.OFFSET_HISTORY['ACTION_SUCCESS']] = 1
            self.obs[self.OFFSET_OBS['HISTORY'] + self.OFFSET_HISTORY['ACTION_SUCCESS'] // 36, 
                     self.OFFSET_HISTORY['ACTION_SUCCESS'] % 36] = 1
        else:
            self.history_features[self.OFFSET_HISTORY['ACTION_SUCCESS'] + 1] = 1
            self.obs[self.OFFSET_OBS['HISTORY'] + (self.OFFSET_HISTORY['ACTION_SUCCESS'] + 1) // 36, 
                     (self.OFFSET_HISTORY['ACTION_SUCCESS'] + 1) % 36] = 1
    
    def _update_round_wind(self, prevalent_wind):
        """
        更新圈风信息
        """
        # 清除之前的圈风信息
        for i in range(4):
            self.history_features[self.OFFSET_HISTORY['ROUND_WIND'] + i] = 0
        
        # 设置新的圈风信息
        self.history_features[self.OFFSET_HISTORY['ROUND_WIND'] + prevalent_wind] = 1
        
        # 更新观察状态中的圈风信息
        for i in range(4):
            idx = self.OFFSET_HISTORY['ROUND_WIND'] + i
            self.obs[self.OFFSET_OBS['HISTORY'] + idx // 36, idx % 36] = self.history_features[idx]
    
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
    
    def _obs(self):
        """
        生成当前观测信息，以及可执行动作的掩码
        """
        # 创建动作掩码
        action_mask = np.zeros(self.ACT_SIZE)
        for act in self.valid:
            action_mask[act] = 1
            
        # 返回观察状态和动作掩码
        return {
            'observation': self.obs.copy(),
            'action_mask': action_mask
        }
        
    def _hand_embedding_update(self):
        """
        根据 self.hand 更新 self.obs 矩阵中的手牌信息部分，以反映当前玩家手中的牌。
        前接 deal
        """
        # 清空手牌信息部分
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HISTORY']] = 0
        d = defaultdict(int) # d[tile] 表示手牌中牌 tile 的数量，默认值为 0
        # 统计手牌中各种牌的数量
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            # 更新 self.obs 矩阵中的手牌信息部分
            # OFFSET_OBS['HAND'] 是常量偏移 2，用于锁定手牌信息部分的起始位置
            # 等价于 self.obs[2:2+d[tile]]
            # 所以这里可以看出来，后面四个维度实际记录的是你拥有几张牌的信息，拥有 3 张牌则 self.obs[2:5] 都是 1
            # 也就是拥有几张就顺序记录几个维度 
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
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