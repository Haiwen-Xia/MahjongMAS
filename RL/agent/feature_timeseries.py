from agent.base_agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgentTimeSeries(MahjongGBAgent):
    OBS_SIZE = 95
    ACT_SIZE = 235
    
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ] # 34 tiles
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}

    NONE_TILE_STR = "None"
    TILES_FOR_FEATURE = TILE_LIST + [NONE_TILE_STR]
    TILE_TO_IDX_FEATURE = {tile: i for i, tile in enumerate(TILES_FOR_FEATURE)}
    NONE_TILE_IDX_FEATURE = TILE_TO_IDX_FEATURE[NONE_TILE_STR]

    ACTION_TYPES_DEF = {
        'QUAN_INFO': 0,
        'INITIAL_HAND': 1,
        'SELF_DRAW_TILE': 2,
        'OTHER_PLAYER_DRAW': 3,
        'PLAY_TILE': 4,
        'PENG': 5,
        'CHI': 6,
        'GANG': 7,
        'BUGANG': 8,
        'WIN': 9,
        'NO_ACTION': 10
    }
    NUM_ACTION_TYPES = 11
    
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

    def __init__(self, seatWind):
        self.seatWind = seatWind 
        self.packs = [[] for _ in range(4)] 
        self.history = [[] for _ in range(4)] 
        self.tileWall = [21] * 4
        
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        
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

    def _reset_current_event_info(self):
        """Resets event-specific information at the start of each request processing."""
        self.current_event_player_relative = -1 
        self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
        self.current_event_card1_str = self.NONE_TILE_STR
        self.current_event_card2_str = self.NONE_TILE_STR

    def request2obs(self, request: str):
        self._reset_current_event_info()
        
        t = request.split()
        action_type_str = t[0]

        if action_type_str == 'Wind':
            self.prevalentWind = int(t[1])
            self.current_event_player_relative = -1 # System event
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['QUAN_INFO']
            return # No observation generated, just updates internal state

        elif action_type_str == 'Deal':
            self.hand = list(t[1:]) 
            for tile_h in self.hand:
                self.shownTiles[tile_h] += 1
            
            self.current_event_player_relative = 0
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['INITIAL_HAND']
            # Card1, Card2 remain NONE_TILE_STR (hand details are internal state, not single cards)
            return # No observation generated
        
        elif action_type_str == 'Huang': # Draw game (no winner)
            self.current_event_player_relative = -1 # System event
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION'] 
            self.valid = []
            return self._obs()

        elif action_type_str == 'Draw':
            if self.tileWall[self.seatWind] > 0:
                self.tileWall[self.seatWind] -= 1
            
            is_current_wall_empty_for_player = (self.tileWall[self.seatWind] == 0)
            # self.wallLast could be set more globally, e.g., if sum(self.tileWall) == 0

            drawn_tile = t[1]
            self.curTile = drawn_tile

            # 初始置空可用操作空间
            self.valid = []

            self.hand.append(drawn_tile)
            self.shownTiles[drawn_tile] += 1 
            
            self.isAboutKong = False # Reset after any draw
            
            if self._check_mahjong(drawn_tile, isSelfDrawn=True, isAboutKong=self.isAboutKong, isWallLast=is_current_wall_empty_for_player):
                self.valid.append(self.OFFSET_ACT['Hu'])
            
            for tile_h_set in set(self.hand): # Play options
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile_h_set])
            
            can_draw_replacement = self.tileWall[self.seatWind] > 0 # Can I draw after Gang/BuGang?
            if can_draw_replacement : 
                for tile_h_set in set(self.hand): # AnGang options
                    if self.hand.count(tile_h_set) == 4:
                        self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile_h_set])
            
                for pack_type, pack_tile, _ in self.packs[self.seatWind]: # BuGang options
                    if pack_type == 'PENG' and pack_tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[pack_tile])
            
            self.current_event_player_relative = 0
            self.current_event_action_type_idx = self.ACTION_TYPES_DEF['SELF_DRAW_TILE']
            self.current_event_card1_str = drawn_tile
            # Card2 remains NONE_TILE_STR
            return self._obs()

        elif action_type_str == 'Player':
            p_abs = int(t[1]) # Absolute player index from input
            p_rel = (p_abs + 4 - self.seatWind) % 4 # Player index relative to self
            event_str = t[2]
            
            self.current_event_player_relative = p_rel

            if event_str == 'Draw': # Other player draws
                if self.tileWall[p_abs] > 0:
                    self.tileWall[p_abs] -= 1
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['OTHER_PLAYER_DRAW']
                # Card1, Card2 remain NONE_TILE_STR
                return # No observation for self

            elif event_str == 'Play':
                played_tile = t[3]
                self.curTile = played_tile # Store for potential actions by self (Peng, Chi, Gang, Hu)
                self.tileFrom = p_rel    # Store who played it (relative index)
                
                self.shownTiles[played_tile] += 1 
                self.history[p_abs].append(played_tile)

                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['PLAY_TILE']
                self.current_event_card1_str = played_tile
                # Card2 remains NONE_TILE_STR

                if p_rel == 0: # Self played a card
                    if played_tile in self.hand: self.hand.remove(played_tile)
                    return # Action completed, no decision for self, so no observation
                else: # Other player played, self might act
                    is_wall_last_for_hu_check = (sum(self.tileWall) == 0) # Or more specific like tileWall[p_abs]==0

                    if self._check_mahjong(played_tile, isSelfDrawn=False, isAboutKong=False, isWallLast=is_wall_last_for_hu_check):
                        self.valid.append(self.OFFSET_ACT['Hu'])
                    
                    can_self_draw_replacement = self.tileWall[self.seatWind] > 0

                    if can_self_draw_replacement: # Peng or Gang on discard only if can draw replacement
                        if self.hand.count(played_tile) >= 2: # Peng
                            self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[played_tile])
                        if self.hand.count(played_tile) == 3: # Gang (MingGang a discard)
                             self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[played_tile])
                    
                    if p_rel == 3 and played_tile[0] in 'WTB' and can_self_draw_replacement: # Chi from upper house
                        color = played_tile[0]
                        num = int(played_tile[1])
                        # Chi actions: middle tile is used for indexing the action
                        # Type 0 (eat left): hand has (num+1, num+2), eats num. Middle is num+1.
                        if num <= 7 and (color+str(num+1)) in self.hand and (color+str(num+2)) in self.hand:
                            middle_tile_str = color + str(num+1)
                            if middle_tile_str in self.OFFSET_TILE: # Ensure tile exists
                                 self.valid.append(self.OFFSET_ACT['Chi'] + self.OFFSET_TILE[middle_tile_str]*3 + 0)
                        # Type 1 (eat middle): hand has (num-1, num+1), eats num. Middle is num.
                        if num >=2 and num <=8 and (color+str(num-1)) in self.hand and (color+str(num+1)) in self.hand:
                            middle_tile_str = color + str(num)
                            if middle_tile_str in self.OFFSET_TILE:
                                self.valid.append(self.OFFSET_ACT['Chi'] + self.OFFSET_TILE[middle_tile_str]*3 + 1)
                        # Type 2 (eat right): hand has (num-2, num-1), eats num. Middle is num-1.
                        if num >= 3 and (color + str(num-2)) in self.hand and (color + str(num-1)) in self.hand:
                            middle_tile_str = color + str(num-1)
                            if middle_tile_str in self.OFFSET_TILE:
                                self.valid.append(self.OFFSET_ACT['Chi'] + self.OFFSET_TILE[middle_tile_str]*3 + 2)
                    
                    self.valid.append(self.OFFSET_ACT['Pass'])
                    return self._obs()

            elif event_str == 'Peng':
                peng_tile = self.curTile # Tile being Peng-ed (set by previous Play event)
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['PENG']
                self.current_event_card1_str = peng_tile
                # Card2 remains NONE_TILE_STR
                
                offer_player_abs = (self.seatWind + self.tileFrom) % 4 # Absolute index of player who offered the tile
                offer_relative_to_p_abs = (offer_player_abs + 4 - p_abs) % 4 # Offer relative to p_abs
                self.packs[p_abs].append(('PENG', peng_tile, offer_relative_to_p_abs))
                self.shownTiles[peng_tile] += 2 # Original discard shown + 2 for peng pack

                if p_rel == 0: # Self Peng-ed
                    for _ in range(2): 
                        if peng_tile in self.hand: self.hand.remove(peng_tile)
                    for tile_h_set in set(self.hand): # Valid actions: Play a tile
                        self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile_h_set])
                    return self._obs() # Self needs to decide which tile to play
                else: # Other player Peng-ed
                    return # No observation for self

            elif event_str == 'Chi':
                middle_tile_chi = t[3] # Middle tile of the Chi sequence
                eaten_tile = self.curTile # Tile that was Chi-ed (from previous Play event)

                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['CHI']
                self.current_event_card1_str = middle_tile_chi
                self.current_event_card2_str = eaten_tile # Eaten tile is secondary info for Chi event

                m_val = int(middle_tile_chi[1])
                e_val = int(eaten_tile[1])
                offer = 0 # 1: eats left, 2: eats middle, 3: eats right (relative to middle_tile_chi)
                if e_val == m_val - 1: offer = 1
                elif e_val == m_val: offer = 2
                elif e_val == m_val + 1: offer = 3
                self.packs[p_abs].append(('CHI', middle_tile_chi, offer))
                
                self.shownTiles[eaten_tile] -=1 # Was shown as discard, now part of pack
                m_suit = middle_tile_chi[0] # Update shown tiles for the whole sequence
                self.shownTiles[m_suit+str(m_val-1)] +=1
                self.shownTiles[m_suit+str(m_val)] +=1
                self.shownTiles[m_suit+str(m_val+1)] +=1

                if p_rel == 0: # Self Chi-ed
                    tiles_to_remove_from_hand = []
                    if offer == 1: tiles_to_remove_from_hand = [m_suit+str(m_val), m_suit+str(m_val+1)]
                    elif offer == 2: tiles_to_remove_from_hand = [m_suit+str(m_val-1), m_suit+str(m_val+1)]
                    elif offer == 3: tiles_to_remove_from_hand = [m_suit+str(m_val-1), m_suit+str(m_val)]
                    for tile_rem in tiles_to_remove_from_hand:
                        if tile_rem in self.hand: self.hand.remove(tile_rem)
                    
                    for tile_h_set in set(self.hand): # Valid actions: Play a tile
                        self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile_h_set])
                    return self._obs() # Self needs to decide which tile to play
                else: # Other player Chi-ed
                    return # No observation for self
            
            elif event_str == 'UnChi':
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
                middle_tile_chi_str = t[3]
                color = middle_tile_chi_str[0]
                num = int(middle_tile_chi_str[1])
                eaten_tile_original = self.curTile # Should be the tile that was eaten
                original_pack_found_and_removed = False
                for i, pack_info in enumerate(self.packs[p_abs]):
                    if pack_info[0] == 'CHI' and pack_info[1] == middle_tile_chi_str:
                        # To be more robust, we might need to check offer value if multiple Chis of same middle tile
                        self.packs[p_abs].pop(i)
                        original_pack_found_and_removed = True
                        break
                
                if original_pack_found_and_removed:
                    if eaten_tile_original: # Should always be true if logic is consistent
                         self.shownTiles[eaten_tile_original] += 1
                    for i in range(-1, 2):
                        self.shownTiles[color + str(num + i)] -= 1
                    
                    if p_rel == 0: # If self is un-chi-ing
                        # feature.py: for i in range(-1, 2): self.hand.append(color + str(num + i))
                        # feature.py: self.hand.remove(self.curTile)
                        for i in range(-1, 2):
                            self.hand.append(color + str(num + i))
                        if eaten_tile_original and eaten_tile_original in self.hand:
                             self.hand.remove(eaten_tile_original)
                return
            
            elif event_str == 'UnPeng':
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
                peng_tile_original = self.curTile # Should be the tile that was Peng-ed

                original_pack_found_and_removed = False
                for i, pack_info in enumerate(self.packs[p_abs]):
                    if pack_info[0] == 'PENG' and pack_info[1] == peng_tile_original:
                        self.packs[p_abs].pop(i)
                        original_pack_found_and_removed = True
                        break
                
                if original_pack_found_and_removed and peng_tile_original:
                    self.shownTiles[peng_tile_original] -= 2
                    if p_rel == 0: # If self is un-peng-ing
                        for _ in range(2):
                            self.hand.append(peng_tile_original)
                return

            elif event_str == 'Gang' or event_str == 'AnGang' or event_str == 'BuGang':
                if self.tileWall[p_abs] > 0: # The player who ganged draws a replacement tile
                    self.tileWall[p_abs] -=1
                
                self.isAboutKong = (p_rel == 0) # If self ganged, set flag for potential Kong-related fans on next draw

                ganged_tile_for_event = self.NONE_TILE_STR # Default for Card1 if tile is concealed

                if event_str == 'AnGang':
                    self.current_event_action_type_idx = self.ACTION_TYPES_DEF['GANG']
                    if p_rel == 0 and len(t) == 4: # Self AnGang, tile is specified "Player N(me) AnGang XX"
                        ganged_tile_str = t[3]
                        ganged_tile_for_event = ganged_tile_str
                        for _ in range(4):
                            if ganged_tile_str in self.hand: self.hand.remove(ganged_tile_str)
                        self.packs[p_abs].append(('GANG', ganged_tile_str, 0)) # Offer 0 for AnGang
                        self.shownTiles[ganged_tile_str] += 4 
                    else: # Other player AnGang ("Player N AnGang"), tile is concealed
                        self.packs[p_abs].append(('GANG', "CONCEALED", 0)) 
                        # ganged_tile_for_event remains NONE_TILE_STR as tile is unknown to others
                    self.current_event_card1_str = ganged_tile_for_event
                
                elif event_str == 'Gang': # MingGang on a discard
                    self.current_event_action_type_idx = self.ACTION_TYPES_DEF['GANG']
                    ganged_tile = self.curTile # Tile from previous Play event
                    ganged_tile_for_event = ganged_tile
                    
                    offer_player_abs_mg = (self.seatWind + self.tileFrom) % 4
                    offer_relative_to_p_abs_mg = (offer_player_abs_mg + 4 - p_abs) % 4
                    self.packs[p_abs].append(('GANG', ganged_tile, offer_relative_to_p_abs_mg))
                    self.shownTiles[ganged_tile] += 3 # Already 1 from discard, +3 for pack
                    if p_rel == 0: # Self MingGanged
                        for _ in range(3):
                            if ganged_tile in self.hand: self.hand.remove(ganged_tile)
                    self.current_event_card1_str = ganged_tile_for_event
                
                elif event_str == 'BuGang': # BuGang (Promoting a Peng to Gang)
                    self.current_event_action_type_idx = self.ACTION_TYPES_DEF['BUGANG']
                    bugang_tile = t[3]
                    ganged_tile_for_event = bugang_tile

                    for i, (pack_type, tile_in_pack, offer_val) in enumerate(self.packs[p_abs]):
                        if pack_type == 'PENG' and tile_in_pack == bugang_tile:
                            self.packs[p_abs][i] = ('GANG', bugang_tile, offer_val) # Keep original offer
                            break
                    self.shownTiles[bugang_tile] += 1 # From Peng (3) to Gang (4)
                    
                    self.current_event_card1_str = ganged_tile_for_event
                    if p_rel == 0: # Self BuGanged
                        if bugang_tile in self.hand: self.hand.remove(bugang_tile)
                        # No _obs() here, expect Draw event from game engine
                    else: # Other player BuGanged, self might rob the kong
                        is_wall_last_for_rob_check = (sum(self.tileWall) == 0) # Or tileWall[p_abs]==0
                        if self._check_mahjong(bugang_tile, isSelfDrawn=False, isAboutKong=True, isWallLast=is_wall_last_for_rob_check):
                            self.valid.append(self.OFFSET_ACT['Hu'])
                        self.valid.append(self.OFFSET_ACT['Pass'])
                        return self._obs() # Self needs to decide whether to rob
                
                # For Gang actions by self or other (not robbed BuGang), no immediate _obs().
                # Game engine will typically send a Draw event for the ganging player.
                return

            elif event_str == 'Hu':
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['WIN']
                # Winning tile is typically self.curTile (set by Draw, Play, or Gang/BuGang that was Hu'd)
                self.current_event_card1_str = self.curTile if self.curTile else self.NONE_TILE_STR
                self.valid = [] # No more actions after Hu
                return self._obs() # Generate final observation for this event

            elif event_str == 'Invalid': 
                self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
                self.valid = []
                return self._obs()
        
        # Fallback for unhandled requests, though ideally all specified inputs are covered
        # print(f"Warning: Unhandled request format: {request}")
        # self.current_event_action_type_idx = self.ACTION_TYPES_DEF['NO_ACTION']
        # return self._obs()


    def action2response(self, action_idx: int) -> str:
        """Converts an action index from the model to a game engine understandable string."""
        if action_idx == self.OFFSET_ACT['Pass']:
            return 'Pass'
        if action_idx == self.OFFSET_ACT['Hu']:
            return 'Hu'
        
        if self.OFFSET_ACT['Play'] <= action_idx < self.OFFSET_ACT['Chi']:
            tile_offset = action_idx - self.OFFSET_ACT['Play']
            if 0 <= tile_offset < len(self.TILE_LIST):
                return 'Play ' + self.TILE_LIST[tile_offset]
        
        if self.OFFSET_ACT['Chi'] <= action_idx < self.OFFSET_ACT['Peng']:
            val = action_idx - self.OFFSET_ACT['Chi']
            tile_idx_in_list = val // 3 # Index of the middle tile in self.TILE_LIST
            # chi_type = val % 3 # 0: eats left, 1: eats middle, 2: eats right (relative to middle tile)
            if 0 <= tile_idx_in_list < len(self.TILE_LIST):
                middle_tile_str = self.TILE_LIST[tile_idx_in_list]
                return f'Chi {middle_tile_str}' # Engine expects "Chi <middle_tile>"

        if self.OFFSET_ACT['Peng'] <= action_idx < self.OFFSET_ACT['Gang']:
            # Peng action string is just "Peng". The tile being Peng-ed is self.curTile.
            # tile_offset = action_idx - self.OFFSET_ACT['Peng']
            # if 0 <= tile_offset < len(self.TILE_LIST):
            #     return 'Peng ' + self.TILE_LIST[tile_offset] # This was if Peng needs tile arg
            return 'Peng' 
        
        if self.OFFSET_ACT['Gang'] <= action_idx < self.OFFSET_ACT['AnGang']:
            # This is for MingGang on a discard. Action string is "Gang". Tile is self.curTile.
            # tile_offset = action_idx - self.OFFSET_ACT['Gang']
            # if 0 <= tile_offset < len(self.TILE_LIST):
            #    return 'Gang ' + self.TILE_LIST[tile_offset] # This was if Gang needs tile arg
            return 'Gang'

        if self.OFFSET_ACT['AnGang'] <= action_idx < self.OFFSET_ACT['BuGang']:
            # AnGang. Action string is "Gang <tile>"
            tile_offset = action_idx - self.OFFSET_ACT['AnGang']
            if 0 <= tile_offset < len(self.TILE_LIST):
                return 'Gang ' + self.TILE_LIST[tile_offset] 

        if self.OFFSET_ACT['BuGang'] <= action_idx < self.ACT_SIZE:
            # BuGang. Action string is "BuGang <tile>"
            tile_offset = action_idx - self.OFFSET_ACT['BuGang']
            if 0 <= tile_offset < len(self.TILE_LIST):
                return 'BuGang ' + self.TILE_LIST[tile_offset]
        
        # Fallback if action_idx is out of expected range or unmapped
        # print(f"Warning: Unmappable action_idx in action2response: {action_idx}")
        return 'Pass'


    def _obs(self):
        """Constructs the observation vector based on the current event and game state."""
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
        
        # 5. Quan Wind (5 bits one-hot: 4 winds + N/A)
        quan_wind_vec = np.zeros(5)
        # Quan wind is relevant primarily for QUAN_INFO events or if globally known
        if self.current_event_action_type_idx == self.ACTION_TYPES_DEF['QUAN_INFO'] and \
           hasattr(self, 'prevalentWind') and 0 <= self.prevalentWind < 4:
            quan_wind_vec[self.prevalentWind] = 1
        else: # For other events, or if prevalentWind is not set, mark as N/A
            quan_wind_vec[4] = 1 

        # 6. Player Wind (Seat Wind) (5 bits one-hot: 4 winds + N/A)
        player_wind_vec = np.zeros(5) 
        if hasattr(self, 'seatWind') and 0 <= self.seatWind < 4:
            player_wind_vec[self.seatWind] = 1 
        else: # Should not happen after __init__ if seatWind is always set
            player_wind_vec[4] = 1 # N/A

        observation_vec = np.concatenate([
            player_vec, action_type_vec, card1_vec, card2_vec, quan_wind_vec, player_wind_vec
        ])

        # Action Mask
        mask = np.zeros(self.ACT_SIZE)
        # self.valid should contain valid action *indices* based on OFFSET_ACT
        if hasattr(self, 'valid') and self.valid: 
            for a_idx in self.valid:
                if 0 <= a_idx < self.ACT_SIZE: # Ensure index is within bounds
                    mask[a_idx] = 1
        
        return {
            'observation': observation_vec,
            'action_mask': mask
        }
    
    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False, isWallLast=False):
        """Checks if a Mahjong hand is formed with the given winTile."""
        if not winTile or winTile == self.NONE_TILE_STR or winTile == "CONCEALED":
            return False

        try:
            current_hand_for_check = list(self.hand) # Use a copy of the current hand
            
            # shownTiles should be updated *before* this check for the winTile in question.
            # So, if shownTiles[winTile] == 4, it means this winTile is the 4th instance visible.
            is_4th_tile_check = (self.shownTiles[winTile] == 4)

            current_prevalent_wind = self.prevalentWind if hasattr(self, 'prevalentWind') and self.prevalentWind != -1 else 0

            fans = MahjongFanCalculator(
                pack=tuple(self.packs[self.seatWind]), 
                hand=tuple(current_hand_for_check), 
                winTile=winTile,
                flowerCount=0, 
                isSelfDrawn=isSelfDrawn,
                is4thTile=is_4th_tile_check,
                isAboutKong=isAboutKong, 
                isWallLast=isWallLast,   
                seatWind=self.seatWind,
                prevalentWind=current_prevalent_wind,
                verbose=False # Set to True for debugging fan calculations
            )
            fanCnt = 0
            for fanPoint, cnt, _, _ in fans: 
                fanCnt += fanPoint * cnt
            if fanCnt < 8: # Standard 8 fan minimum for Mahjong
                return False
        except Exception: # Catch any errors during fan calculation
            return False
        return True