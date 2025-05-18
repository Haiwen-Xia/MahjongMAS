import numpy as np
import os
from feature_timeseries import FeatureAgentTimeSeries
from preprocess import event_to_string

def parse_raw_text(input_file):
    """从原始文本解析事件序列"""
    lines = []
    with open(input_file, 'r', encoding='UTF-8') as f:
        line = True
        while line:
            line = f.readline()            
            lines.append(line)
    
    # 过滤掉注释行和空行
    raw_lines = []
    cnt = 0
    for line in lines:
        t = line.split()
        line = line.strip()        
        if not t:
             line = ''
        else:
            if  t[0] in ['Fan','Score','Deal','Wind']:    
                continue
            if t[0] == 'Match':
                cnt += 1
                line = "Match #%d\n" % cnt
            if len(t) > 2 and t[2] == 'Deal':
                continue
        if line and not line.startswith('//'):
            raw_lines.append(line)
    
    return raw_lines

def load_history_events(data_dir, match_start=0, match_count=15):
    """加载保存的history数据并转换为可读形式"""
    match_events = []
    match_count = len(os.listdir(data_dir)) // 4  # 每局比赛有4个文件
    for match_idx in range(match_start, match_start + match_count):
        match_events.append(f"Match #{match_idx+1}\n")
        for player in range(1):
            try:
                file_path = os.path.join(data_dir, f"{match_idx*4+player}.npz")
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    
                    # 检查是否有history键
                    if 'history' in data:
                        history = data['history']
                        print( f"{match_idx*4+player}.npz")
                        print(history[0])
                        # 记录此match的事件
                        for event_vec in history:
                            event_str = event_to_string(event_vec, player)
                            match_events.append(event_str)
                    else:
                        print(f"Warning: No 'history' key in file {file_path}")
            except Exception as e:
                print(f"Error loading data for match {match_idx}, player {player}: {e}")
        
        # 根据玩家顺序排序事件
        # 这里我们假设每个玩家的事件在时间上是顺序记录的
        # 我们以玩家0的事件作为主序列，然后按顺序穿插其他玩家的事件
    
    return match_events

def save_groundtruth_events(raw_lines, output_dir):
    """保存原始文本事件到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'groundtruth_events.txt'), 'w', encoding='UTF-8') as f:
        f.write("=============== 原始事件序列 ===============\n\n")
        for line in raw_lines:
            f.write(f"{line}\n")

def save_program_events(events, output_dir):
    """保存程序记录的事件到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'program_events.txt'), 'w', encoding='UTF-8') as f:
        f.write("=============== 程序记录事件序列 ===============\n\n")
        for event in events:
            f.write(f"{event}\n")

def main():
    input_file = 'd:/local/大二下CodeSpace/MAS/Mahjong/IL/sequence_modeling/sample.txt'
    data_dir = 'd:/local/大二下CodeSpace/MAS/Mahjong/IL/sequence_modeling/data'
    output_dir = 'd:/local/大二下CodeSpace/MAS/Mahjong/IL/sequence_modeling/event_comparison'
    
    print("解析原始文本...")
    raw_lines = parse_raw_text(input_file)
    
    print("加载程序记录的事件...")
    program_events = load_history_events(data_dir)
    
    print("保存事件到文件...")
    save_groundtruth_events(raw_lines, output_dir)
    save_program_events(program_events, output_dir)
    
    print(f"文件已保存到 {output_dir} 目录")
    print(f"- groundtruth_events.txt: 原始事件")
    print(f"- program_events.txt: 程序记录事件")

if __name__ == "__main__":
    main()