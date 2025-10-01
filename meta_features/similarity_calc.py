import json
import numpy as np
from tqdm import tqdm

def epanechnikov_similarity(x, y, h=1.0, p=2):
    """
    Epanechnikov quadratic kernel similarity
    x, y: 向量
    h: 带宽
    p: 距离的范数 (默认 L2)
    """
    dist = np.linalg.norm(np.array(x) - np.array(y), ord=p) / h
    if dist <= 1:
        return 0.75 * (1 - dist**2)
    else:
        return 0.0

def process_jsonl(input_file, output_file):
    # 读取所有记录
    with open(input_file, 'r') as f:
        records = [json.loads(line) for line in f]
    
    results = []
    
    # 对每条记录进行处理
    for current_record in tqdm(records, desc="Processing records"):
        current_model = current_record['model_name']
        current_hardware = current_record['hardware']
        current_vector = current_record['vector']
        
        similarities = []
        
        # 与其他记录比较
        for other_record in records:
            if (other_record['model_name'] == current_model and 
                other_record['hardware'] != current_hardware):
                
                similarity = epanechnikov_similarity(
                    current_vector, 
                    other_record['vector']
                )
                
                similarities.append({
                    'similarity': similarity,
                    'model': other_record['model_name'],
                    'hardware': other_record['hardware']
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 只保留最高相似度的结果
        if similarities:
            best_match = similarities[0]
            results.append({
                'model': current_model,
                'hardware': current_hardware,
                'rank1_similarity': best_match['similarity'],
                'rank1_model': best_match['model'],
                'rank1_hardware': best_match['hardware']
            })
        else:
            results.append({
                'model': current_model,
                'hardware': current_hardware,
                'rank1_similarity': None,
                'rank1_model': None,
                'rank1_hardware': None
            })
    
    # 写入结果到新的jsonl文件
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

# 使用示例
input_file = 'records.jsonl'
output_file = 'similarity_records.jsonl'
process_jsonl(input_file, output_file)