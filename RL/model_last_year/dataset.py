from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right
import random

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False):
        import json
        with open('autodl-tmp/data/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        
        # 以下代码会在初始化时一次性读入所有数据，并且储存在cache内，每次取数据时用索引从cache中取出，但这样对内存开销太大 
        # self.cache = {'obs': [], 'mask': [], 'act': []}
        # for i in range(self.matches):
        #     if i % 128 == 0: print('loading', i)
        #     d = np.load('data/%d.npz' % (i + self.begin))
        #     for k in d:
        #         self.cache[k].append(d[k])
    
    def __len__(self):
        return self.samples
        

    def transform(self,obs,act,mask):
        
        tp = np.zeros(235)
        tp[act] = 1
        act = tp
        
        prob_threshold = 0.5
        
        head1 = act[0:2]
        head2 = mask[0:2]
        
        play1 = np.reshape(act[2:29],(3,9))
        ming1 = np.reshape(act[36:63],(3,9))
        an1 = np.reshape(act[70:97],(3,9))
        bu1 = np.reshape(act[104:131],(3,9))
        peng1 = np.reshape(act[138:165],(3,9))
        chi1 = np.reshape(act[172:235],(3,7,3))
        
        play2 = np.reshape(mask[2:29],(3,9))
        ming2 = np.reshape(mask[36:63],(3,9))
        an2 = np.reshape(mask[70:97],(3,9))
        bu2 = np.reshape(mask[104:131],(3,9))
        peng2 = np.reshape(mask[138:165],(3,9))
        chi2 = np.reshape(mask[172:235],(3,7,3))
        
        dic = [[0,8],[1,7],[2,6],[3,5]]
        rand_num = random.random()
        if rand_num < prob_threshold:
        # 选择要交换的列的索引
            for i in range(4):
                col1, col2 = dic[i]
                obs[:198,:,[col1,col2]] = obs[:198,:,[col2,col1]]
                play1[:,[col1 ,col2 ]] = play1[:,[col2 ,col1 ]]
                play2[:,[col1 ,col2 ]] = play2[:,[col2 ,col1 ]]
                ming1[:,[col1 ,col2 ]] = ming1[:,[col2 ,col1 ]]
                ming2[:,[col1 ,col2 ]] = ming2[:,[col2 ,col1 ]]
                an1[:,[col1 ,col2 ]] = an1[:,[col2 ,col1 ]]
                an2[:,[col1 ,col2 ]] = an2[:,[col2 ,col1 ]]
                bu1[:,[col1 ,col2 ]] = bu1[:,[col2 ,col1 ]]
                bu2[:,[col1 ,col2 ]] = bu2[:,[col2 ,col1 ]]
                peng1[:,[col1 ,col2 ]] = peng1[:,[col2 ,col1 ]]
                peng2[:,[col1 ,col2 ]] = peng2[:,[col2 ,col1 ]]                    
                if i >= 1:
                    chi1[:,[col1 - 1,col2 - 1],:] = chi1[:,[col2 - 1,col1 - 1],:]
                    chi2[:,[col1 - 1,col2 - 1],:] = chi2[:,[col2 - 1,col1 - 1],:]
            chi1[:,:,[0,2]] = chi1[:,:,[2,0]]
            chi2[:,:,[0,2]] = chi2[:,:,[2,0]]

        dic = [[0,1],[0,2],[1,2]]
        
        prob_threshold = 0.2
        for i in range(3):
            rand_num = random.random()
            if rand_num < prob_threshold:
                row1, row2 = dic[i]
                obs[:198,[row1,row2],:] = obs[:198,[row2,row1],:]
                play1[[row1,row2],:] = play1[[row2,row1],:]
                play2[[row1,row2],:] = play2[[row2,row1],:]
                ming1[[row1,row2],:] = ming1[[row2,row1],:]
                ming2[[row1,row2],:] = ming2[[row2,row1],:]
                an1[[row1,row2],:] = an1[[row2,row1],:]
                an2[[row1,row2],:] = an2[[row2,row1],:]
                bu1[[row1,row2],:] = bu1[[row2,row1],:]
                bu2[[row1,row2],:] = bu2[[row2,row1],:]
                peng1[[row1,row2],:] = peng1[[row2,row1],:]
                peng2[[row1,row2],:] = peng2[[row2,row1],:]
                
                chi1[[row1,row2],:,:] = chi1[[row2,row1],:,:]
                chi2[[row1,row2],:,:] = chi2[[row2,row1],:,:]

        new_act = np.concatenate((head1 , play1.reshape(27) , act[29:36] , ming1.reshape(27) , act[63:70] , an1.reshape(27) , act[97:104] , bu1.reshape(27) , act[131:138] , peng1.reshape(27) , act[165:172] , chi1.reshape(63)))
        new_mask = np.concatenate((head2 , play2.reshape(27) , mask[29:36] , ming2.reshape(27) , mask[63:70] , an2.reshape(27) , mask[97:104] , bu2.reshape(27) , mask[131:138] , peng2.reshape(27) , mask[165:172] , chi2.reshape(63)))
        new_act = np.argmax(new_act)
        return obs,new_act,new_mask
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        
        # 每次取出部分数据时才加载这部分数据
        d = np.load('autodl-tmp/data/%d.npz' % (match_id + self.begin))
        
        # 如果需要数据加强
        if self.augment == True:
            
            d['obs'][sample_id],d['act'][sample_id],d['mask'][sample_id] = self.transform(d['obs'][sample_id],d['act'][sample_id],d['mask'][sample_id])
            
            d['act'][sample_id] = np.argmax(d['act'][sample_id])
            
        return (d['obs'][sample_id], d['mask'][sample_id], d['act'][sample_id])