建模历史，粒度本身就是一个可粗可细的问题；我觉得粗一点可能就足够了？
粒度按照data.txt(当然ignore要分开)就可以了，
pass作为一个action【需要决策pass与否的位置肯定是需要的】,但不作为我们历史考虑的范畴？
// pass什么时候作为一个合法动作在preprocess.py里面的action.pop里写了，逻辑还挺绕的,
// preprocess那里面的actions也是一个“近似时序”的东西, action[0][i]不一定比action[1][i+1]早

格式：
仿照圣遗物，每一次对局存一个文件，包含完整对局信息，类似
```python 
    np.savez('data/%d.npz'%matchid
        , history = np.stack([x['history'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , hand = np.stack([x['hand'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , mask = np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , act = np.array([x for i in range(4) for x in actions[i]])
    )
```
以及一个json文件来说明一些格式，比如到底有多少特征维度，多少entry...(preprocess里好像也有类似的)