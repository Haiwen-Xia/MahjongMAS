
from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


# FeatureAgent 的定位是什么样的?

#######################################################################################################################
# 比赛平台 ----> __main__.py ---(request)---> FeatureAgent ---(obs)---> Net(模型) ---(action)---> FeatureAgent ---(request)---> __main__.py ----> 比赛平台#
#######################################################################################################################

# 1. FeatureAgent 负责接受从收集的信息中提取特征, 将其记忆下来, 并且整合为模型容易处理的形式
# 2. 简单来说, (按照目前的设计) 我们的模型不负责记忆任何对局信息, 只负责基于 FeatureAgent 提供的整合信息来做决策
# 3. 我们的 FeatureAgent 除了要提供 obs 以外, 还要提供 action_mask, 用来滤除不合法的动作.
# 4. FeatureAgent 可以理解为一个从原始信息到 obs 的映射, 不过这个映射是带有记忆的.
# 5. FeatureAgent 还要负责解析动作, 将模型的输出转换为平台可以理解的动作, 不过我们基本不会改变这一部分的设计

class FeatureAgent(MahjongGBAgent):
    # 我们整合特征的方式是将其放在一个 OBS_SIZE 通道, 宽高为 9 * 4 的特征图中, 我们主要考虑修改这一部分特征的处理方式
    OBS_SIZE = 187

    # 我们还需要提供一个 action_mask, 由于设计的动作维度是 235, 所以 action_mask 的大小是 235, 不过我们应该不会改变这一部分
    ACT_SIZE = 235
    
    # 为了让我们修改特征图时不需要考虑具体的通道索引, 而是考虑具体的特征名称, 我们定义了一个偏移量的字典
    OFFSET_OBS = {
        # ===== 基础信息: 座风, 场风, 己方手牌 =====
        'SEAT_WIND' : 0,        # 座风      通道 0
        'PREVALENT_WIND' : 1,   # 场风      通道 1
        'HAND' : 2,             # 己方手牌  通道 2-5 (共 4 个)

        # ===== 特殊操作记录: 所有玩家的吃、碰、杠牌, 弃牌 =====

        # =============== 吃牌 ===============
        # 我们要记录每个玩家的吃牌记录, 对于每个玩家
        # 从前往后记录, 显然每个人最多吃 4 次, 每一次操作占 2 个通道, 
        # 第一个通道给出顺子, 第二个通道标出被吃的牌, 强化关联性
        'CHI0' : 6,             # 己方吃牌 通道 6 - 13 (共 8 个) 
        'CHI1' : 14,            # 下家吃牌 通道 14 - 21 (共 8 个)
        'CHI2' : 22,            # 对家吃牌 通道 22 - 29 (共 8 个)
        'CHI3' : 30,            # 上家吃牌 通道 30 - 37 (共 8 个)

        # =============== 碰牌 ===============
        # 我们要记录每个玩家的碰牌记录, 对于每个玩家
        # 并非从前往后记录, 而是记录在一个通道内, 从前向后依次记录了碰牌的来源
        # 第一个通道对应于来自 上家
        # 第二个通道对应于来自 对家
        # 第三个通道对应于来自 下家
        # 碰牌不可能来自于自己, 因此我们需要预留 3 个通道 每玩家
        'PENG0' : 38,           # 己方碰牌 通道 38 - 40 (共 3 个) 
        'PENG1' : 41,           # 下家碰牌 通道 41 - 43 (共 3 个)
        'PENG2' : 44,           # 对家碰牌 通道 44 - 46 (共 3 个)
        'PENG3' : 47,           # 上家碰牌 通道 47 - 49 (共 3 个)

        # ============== 杠牌 ===============
        # 我们要记录每个玩家的杠牌记录, 对于每个玩家
        # 我们将杠牌记录在同一个通道内, 从前向后依次记录了杠牌的来源
        # 第一个通道对应于来自 自己
        # 第二个通道对应于来自 上家
        # 第三个通道对应于来自 对家
        # 第四个通道对应于来自 下家
        'GANG0' : 50,           # 己方杠牌 通道 50 - 53 (共 4 个)
        'GANG1' : 54,           # 下家杠牌 通道 54 - 57 (共 4 个)
        'GANG2' : 58,           # 对家杠牌 通道 58 - 61 (共 4 个)
        'GANG3' : 62,           # 上家杠牌 通道 62 - 65 (共 4 个)

        # ============== 弃牌 ===============
        # 记录每个玩家的弃牌信息, 对于每个玩家, 其最多弃 28 张牌
        'PLAY0' : 63,           # 己方弃牌 通道 63 - 90 (共 28 个)
        'PLAY1' : 91,           # 下家弃牌 通道 91 - 118 (共 28 个)
        'PLAY2' : 119,          # 对家弃牌 通道 119 - 146 (共 28 个)
        'PLAY3' : 147,          # 上家弃牌 通道 147 - 174 (共 28 个)

        # ============== 明牌 ===============
        # 记录目前已经展示出来的牌的数量, 例如 吃碰杠以及弃牌
        'SHOW': 175,            # 明牌数量 通道 175 - 178 (共 4 个)

        # ===== 其他信息 =====
        # ============== 牌墙剩余数量 ===========
        'REMAIN0' : 179,
        'REMAIN1' : 180,
        'REMAIN2' : 181,
        'REMAIN3' : 182,

        # ============== 暗杠 ===============
        # 我们要记录每个玩家暗杠了多少次, 由于通常暗杠的次数都很少, 不妨记录在
        'ANGANG0' : 183,          # 己方暗杠 通道 183
        'ANGANG1' : 184,          # 下家暗杠 通道 184
        'ANGANG2' : 185,          # 对家暗杠 通道 185
        'ANGANG3' : 186,          # 上家暗杠 通道 186
    }

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
        
        # 初始化一个 obs 张量, 这将被用来存储信息, 它的大小是 6 * 4 * 9, 并输入给模型
        self.obs = np.zeros((self.OBS_SIZE, 36))

        # 更新观测信息中的座风信息
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
    
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
            # 更新观测信息中的场风信息, 因为这个长期不变, 所以我们只需要在这里更新一次
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1

            # 不需要返回任何值, 因为当接受到 `Wind` 时, __main__.py 中会代替我们输出 `PASS`
            return
        
        # 比赛开始, 发牌阶段, 由于原先什么牌也没有, 所以直接用其来初始化一个 self.hand
        if t[0] == 'Deal':
            """
            Deal 牌1 牌2 ...
            """
            self.hand = t[1:]
            # 下面这个函数会根据 self.hand 更新 self.obs
            self._hand_embedding_update()

            return
        
        # 荒庄 (没有牌可摸)
        if t[0] == 'Huang':
            """
            Huang
            """
            # 无可用操作
            self.valid = []
            # 返回信息
            return self._obs()
        

        # 摸牌
        if t[0] == 'Draw':
            """
            Draw 牌1
            """
            # 可选动作： Hu, Play, AnGang, BuGang
            
            # 减少牌墙数量
            self.tileWall[0] -= 1
            
            # 检查下家是否是最后一张牌
            self.wallLast = self.tileWall[1] == 0

            # 获取摸到的牌
            tile = t[1]

            # 初始置空可用操作空间
            self.valid = []
            
            # 检查是否能胡牌 (通过封装的算番函数)
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
        p = (int(t[1]) + 4 - self.seatWind) % 4
        
        # Player N Draw
        if t[2] == 'Draw':
            """
            玩家 p 摸牌
            """
            # 减少玩家 p 的牌墙数量
            self.tileWall[p] -= 1

            # 检查下家是否是最后一张牌, 因为这个时候当前玩家要弃牌, 如果下家牌堆没有牌了, 那么就可能触发 [妙手回春/海底捞月]
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return

        # Player N Invalid
        if t[2] == 'Invalid':
            """
            玩家 p 的操作无效
            """
            # 初始置空可用操作空间
            self.valid = []
            return self._obs()

        # Player N Hu
        if t[2] == 'Hu':
            """
            玩家 p 胡牌
            """
            # 初始置空可用操作空间
            self.valid = []
            return self._obs()

        # Player N Play XX [其他玩家弃牌了]
        if t[2] == 'Play':
            """
            玩家 p 打出一张牌 XX
            """
            # 获取打出的牌
            self.tileFrom = p
            self.curTile = t[3]

            # 更新打出的牌数量 (这是一个全局的字典, 例如目前场上一共打出了多少张 五万)
            self.shownTiles[self.curTile] += 1

            # 将打出的牌记录在 [这个用户的] 历史记录中
            self.history[p].append(self.curTile)
            
            # 如果是自己打出的牌, 那么什么也不用做
            if p == 0:
                # 从手牌中移除打出的牌，更新手牌
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            
            # 如果是别人打出了某张牌, 那么就需要我们考虑 吃/碰/杠/胡/过 等操作
            else:
                # 可选动作：Hu, Gang, Peng, Chi, Pass
                self.valid = []

                # 检查是否能胡牌
                if self._check_mahjong(self.curTile):
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
            # 这是顺子的 **中心牌**, 未必是被吃的牌
            tile = t[3]
            color = tile[0]

            # 这是中心牌的序号 (例如顺子 W4, W5, W6 中的 W5)
            num = int(tile[1])

            # 更新对应玩家吃牌的记录
            ## 这里的 curTile 为什么不变呢?
            ## 这是因为 吃 一定发生在某个玩家弃牌后, 在弃牌时我们已经更新过了 curTile, 因此我们可以直接使用它

            # 这里的 self.curTile[1] 存储了被吃的牌的序号 (例如被吃的是 W6, 形成顺子 W4, W5, W6)
            # 这里记录的第一个是 行为, 第二个是 中心牌, 第三个是被吃的牌在顺子中的位置, i = 1, 2, 3
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))

            # 之前我们在弃牌的时候将 self.shownTiles[self.curTile] + 1, 意思是弃牌堆中某种牌的数量 + 1
            # 但是现在它落到了玩家 p 的手牌中, 变成了不可见的, 所以我们要将它 -1
            self.shownTiles[self.curTile] -= 1

            # 形成的顺子会被展示出来
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1


            # 检查下家是否是最后一张牌, 因为吃完后要弃牌, 上家弃牌可能会触发 [妙手回春/海底捞月]
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            # 如果收到的 request 是自己吃了牌 [注意这里只是假设自己吃成功了!]
            # 那么我们要考虑弃什么牌
            if p == 0:
                
                # 首先我们假装吃成功了, 然后我们要从手牌中移除被吃的牌
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()

                # 考虑弃牌, 手上的牌都可以弃
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        # Player N UnChi XX
        # 这仅仅发生在自己先假设自己吃了牌, 但是最终平台告诉你吃牌失败了 (因为杠/碰/胡优先级更高)
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

            # 总之需要还原, 具体不用管
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
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
            # 这里第一个是 行为, 第二个是 碰牌, 第三个则是相对于碰牌者的相对位置, 上家是 1, 对家是 2, 下家是 3
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))

            self.shownTiles[self.curTile] += 2

            # 检查是否是最后一张牌, 类似地, 考虑下家是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            # 同样还是假设自己碰成功了, 然后我们要从手牌中移除被吃的牌
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
            # 和碰差不多, 上家是 1, 对家是 2, 下家是 3
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3

            # 假设自己杠成功了, 然后我们要从手牌中移除被吃的牌
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

            # 如果是自己暗杠, 才知道是什么牌
            # 如果其他人暗杠了, 那么我们只知道是暗杠, 但是不知道是什么牌
            tile = 'CONCEALED' if p else t[3]

            # 更新暗杠的记录
            self.packs[p].append(('GANG', tile, 0))

            # 在观测矩阵中记录暗杠次数
            for i in range(36):
                self.obs[self.OFFSET_OBS['ANGANG%d' % p]][i] = self.obs[self.OFFSET_OBS['ANGANG%d' % p]][i] + 1

            if p == 0:
                self.isAboutKong = True
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
                # 在三元组中找到对应的牌, 这里的 tile 是补杠的牌
                # 因为既然能够补杠, 说明其一定是碰过的牌, 已经有 3 张了, 而还能杠, 说明不存在位于顺子的可能, 故可以这样处理
                if tile == self.packs[p][i][1]:
                    # 其他不变, 仅仅是把 'PENG' 改为 'GANG'
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            

            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
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
    
    # 这是我们构造观测信息的函数, 是我们代码的核心, 之后我会仔细检查实现的细节
    def _obs(self):
        """
        生成当前观测信息，以及可执行动作的掩码
        """

        # 座风场风信息不变, 清空之后的所有信息, 并且更新手牌
        self._hand_embedding_update()

        # 考虑所有记录的动作
        self._operation_embedding_update()
        
        # 添加弃牌历史
        self._playhistory_embedding_update()

        # 添加展示出来的牌
        self._shown_tiles_embedding_update()

        # 添加剩余牌数
        self._remains_embedding_update()


        # 设置 mask
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            # 存储当前可以执行的动作，通过置 1 来实现
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }
    
    # 一个更新手牌的辅助函数, 可以算作刚才的 _obs 函数的一个子函数, 这个函数会根据手牌更新 obs 中的手牌信息部分
    def _hand_embedding_update(self):
        """
        根据 self.hand 更新 self.obs 矩阵中的手牌信息部分，以反映当前玩家手中的牌。
        前接 deal
        """
        # 清空手牌之后到暗杠数量之前的所有信息
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['ANGANG0']] = 0


        # 统计手牌中各种牌的数量
        d = defaultdict(int) # d[tile] 表示手牌中牌 tile 的数量，默认值为 0
        for tile in self.hand:
            d[tile] += 1

        for tile in d:
            # 更新 self.obs 矩阵中的手牌信息部分
            # OFFSET_OBS['HAND'] 是常量偏移 2，用于锁定手牌信息部分的起始位置
            # 等价于 self.obs[2:2+d[tile]]
            # 所以这里可以看出来，后面四个维度实际记录的是你拥有几张牌的信息，拥有 3 张牌则 self.obs[2:5] 都是 1
            # 也就是拥有几张就顺序记录几个维度 
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1

    def _operation_embedding_update(self):
        for p in range(4): # 这里的 p 是玩家编号, 0 是自己, 1 是下家, 2 是对家, 3 是上家

            chi_cnt = 0
            for operate in self.packs[p]:
                act = operate[0]
                tile = operate[1]
                pos = operate[2] 

                if act == 'CHI':
                    # tile 是顺子中心牌
                    # pos = 1, 2, 3, 表示被吃的牌是顺子的第 1, 2, 3 张牌

                    target_channel = self.OFFSET_OBS['CHI%d' % p] + chi_cnt * 2
                    # 每次吃操作占用 2 个通道
                    # 第一个通道给出顺子
                    self.obs[target_channel][self.OFFSET_TILE[tile] - 1 : self.OFFSET_TILE[tile] + 2] = 1
                    # 第二个通道给出吃的牌, 例如 pos = 2, 则说明被吃的是中心牌, 我们恰好在 tile 对应的位置上加 1
                    self.obs[target_channel + 1][self.OFFSET_TILE[tile] + pos - 2] = 1
                    chi_cnt += 1

                elif act == 'PENG':
                    # tile 是碰牌
                    # 相对于玩家 p 的位置, pos = 0 是自己, 1 是上家, 2 是对家, 3 是下家
                    # 在碰的情况下, pos = 1,2,3 
                    
                    self.obs[self.OFFSET_OBS['PENG%d' % p] + pos - 1][self.OFFSET_TILE[tile]] = 1

                elif act == 'GANG':
  
                    if tile == 'CONCEALED':
                        # 如果是暗杠, 那么 pos = 0
                        pass

                    else:
                        # 如果是明杠, 那么 pos = 0, 1, 2, 3
                        # 这里的 pos 是相对于玩家 p 的位置, 0 是自己, 1 是上家, 2 是对家, 3 是下家
                        # 这里的 tile 是杠牌

                        self.obs[self.OFFSET_OBS['GANG%d' % p] + pos][self.OFFSET_TILE[tile]] = 1

                else:
                    raise NotImplementedError('Unknown operation %s!' % operate)


    def _playhistory_embedding_update(self):
        for p in range(4):
            i = 0
            for tile in self.history[p]:
                self.obs[self.OFFSET_OBS['PLAY%d' % p] + i][self.OFFSET_TILE[tile]] = 1
                i = i + 1
        
    def _shown_tiles_embedding_update(self):
        d = defaultdict(int) # 表示展示出来的牌的数量，默认值为 0
        for tile in self.shownTiles:
            d[tile] += 1

        for tile in d:
            self.obs[self.OFFSET_OBS['SHOW'] : self.OFFSET_OBS['SHOW'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _remains_embedding_update(self):
        for p in range(4):
            # 整个通道设置为剩余牌数
            self.obs[self.OFFSET_OBS['REMAIN%d' % p]] = self.tileWall[p] 


            
            
    

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