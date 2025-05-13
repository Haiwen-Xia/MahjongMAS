# 示例测试代码 (使用 unittest)
import unittest
from feature_timeseries import FeatureAgentTimeSeries # 假设你的文件名为 feature_timeseries.py

class TestFeatureAgentTimeSeriesChi(unittest.TestCase):
    def setUp(self):
        self.agent = FeatureAgentTimeSeries(seatWind=0)
        self.agent.request2obs("Wind 0") # Prevalent wind East
        # Initial hand: W1, W3, T1, T2, B1, B2, F1, J1, W7, W8, W9, T7, T8
        # For simplicity, let's assume shownTiles are updated correctly by Deal
        self.agent.hand = ['W1', 'W3', 'T1', 'T2', 'B1', 'B2', 'F1', 'J1', 'W7', 'W8', 'W9', 'T7', 'T8']
        for tile in self.agent.hand:
            self.agent.shownTiles[tile] +=1
        self.agent.current_event_action_type_idx = self.agent.ACTION_TYPES_DEF['INITIAL_HAND'] # Simulate deal state
        print(f"Initial hand: {self.agent.hand}")
        print(f"Initial shown: {dict(self.agent.shownTiles)}")


    def test_self_chi_and_play_event_generation(self):
        # 1. Player 3 (upper house) plays W2
        obs_data_p3_play = self.agent.request2obs("Player 3 Play W2")
        
        # Assertions for observation when Player 3 plays W2
        self.assertEqual(self.agent.current_event_player_relative, 3)
        self.assertEqual(self.agent.current_event_action_type_idx, self.agent.ACTION_TYPES_DEF['PLAY_TILE'])
        self.assertEqual(self.agent.current_event_card1_str, 'W2')
        self.assertEqual(self.agent.current_event_card2_str, self.agent.NONE_TILE_STR)
        
        # Find the action index for Chi W2 (middle tile W2, eating W2 with W1,W3)
        # Type 1: eat middle (W1, W3 in hand, eat W2). Middle tile is W2.
        # Action index for Chi: OFFSET_ACT['Chi'] + TILE_TO_IDX_FEATURE['W2']*3 + 1
        # Note: TILE_TO_IDX_FEATURE includes "None", OFFSET_TILE does not.
        # The Chi action encoding in feature.py was: 'WTB'.index(color) * 21 + (num - 3) * 3 + type
        # The Chi action encoding in feature_timeseries.py (valid actions) is:
        # self.OFFSET_ACT['Chi'] + self.OFFSET_TILE[middle_tile_str]*3 + type
        # Here, middle_tile_str is 'W2'. type is 1 (eat middle).
        expected_chi_action_idx = -1
        if 'W2' in self.agent.OFFSET_TILE:
            idx_W2_in_tile_list = self.agent.OFFSET_TILE['W2']
            expected_chi_action_idx = self.agent.OFFSET_ACT['Chi'] + idx_W2_in_tile_list * 3 + 1 # Eat middle W2
        
        self.assertTrue(obs_data_p3_play['action_mask'][expected_chi_action_idx] == 1)

        # 2. Agent decides to Chi W2. Game engine sends "Player 0 Chi W2"
        # The 'offer' in packs for Chi is relative to the middle tile.
        # If W2 is eaten, and W1,W3 are used, W2 is the middle. Offer is 2.
        # request2obs for "Player 0 Chi W2"
        obs_data_self_chi = self.agent.request2obs("Player 0 Chi W2")

        # Assertions after self Chi W2
        self.assertEqual(self.agent.current_event_player_relative, 0)
        self.assertEqual(self.agent.current_event_action_type_idx, self.agent.ACTION_TYPES_DEF['CHI'])
        self.assertEqual(self.agent.current_event_card1_str, 'W2') # middle_tile_chi
        self.assertEqual(self.agent.current_event_card2_str, 'W2') # eaten_tile (which was self.curTile)

        self.assertNotIn('W1', self.agent.hand)
        self.assertNotIn('W3', self.agent.hand)
        # self.assertIn('W2', self.agent.hand) # W2 was eaten, not added to hand to be played.
        
        self.assertEqual(len(self.agent.hand), 13 - 2) # 13 initial - 2 for chi = 11, then must play one

        found_chi_pack = False
        for pack_type, tile_in_pack, offer_val in self.agent.packs[0]:
            if pack_type == 'CHI' and tile_in_pack == 'W2': # Middle tile is W2
                found_chi_pack = True
                # Offer for (W1,W2,W3) where W2 is eaten:
                # eaten W2, middle W2 -> offer = 2
                self.assertEqual(offer_val, 2) 
                break
        self.assertTrue(found_chi_pack)

        # shownTiles: W2 was played (+1), then eaten (-1 from discard pool), then W1,W2,W3 in pack (+1 each)
        # Initial: W1:1, W3:1. Others 0 or from deal.
        # P3 plays W2: shownTiles['W2'] becomes 1 (or increments if already seen)
        # Self Chi W2:
        #   self.shownTiles[self.curTile(W2)] -= 1 (from discard)
        #   self.shownTiles[W1] += 1
        #   self.shownTiles[W2] += 1 (as part of sequence)
        #   self.shownTiles[W3] += 1
        # So, W1 should be 2, W2 should be 1 (or original_W2+1), W3 should be 2.
        # This needs careful tracking based on initial shownTiles state.
        # For this test, assuming W1,W3 were 1 from hand, W2 was 0 before play.
        # After P3 plays W2: shownTiles['W2'] = 1
        # After self Chi W2 (middle W2, eaten W2):
        #   shownTiles['W2'] -= 1 (becomes 0)
        #   shownTiles['W1'] += 1 (becomes 2)
        #   shownTiles['W2'] += 1 (becomes 1, this is the W2 in the Chi sequence)
        #   shownTiles['W3'] += 1 (becomes 2)
        self.assertEqual(self.agent.shownTiles['W1'], 2)
        self.assertEqual(self.agent.shownTiles['W2'], 1) # Assuming W2 was not otherwise visible before P3 played it
        self.assertEqual(self.agent.shownTiles['W3'], 2)


        # Check valid actions (must be Play actions from remaining hand)
        for tile_in_hand in set(self.agent.hand):
            play_action_idx = self.agent.OFFSET_ACT['Play'] + self.agent.OFFSET_TILE[tile_in_hand]
            self.assertTrue(obs_data_self_chi['action_mask'][play_action_idx] == 1)
        
        # Verify no other types of actions are valid
        self.assertTrue(obs_data_self_chi['action_mask'][self.agent.OFFSET_ACT['Hu']] == 0)
        # ... etc. for Peng, Gang, Pass (Pass might be valid if no tiles to play, but not here)

if __name__ == '__main__':
    unittest.main()