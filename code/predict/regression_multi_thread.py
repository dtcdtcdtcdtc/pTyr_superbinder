import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

# 配置GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 对所有GPU设备统一设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 加载路径配置（需根据实际路径修改）
model_save_path = "model/CNN_regression_model/"  # 替换为实际模型路径
standard_file_path = "standard_regression_file/SH2_R24_SUM_fre_regression.csv"  # 替换为标准数据文件路径

# 初始化标准化器（与训练时一致）
data = pd.read_csv(standard_file_path, sep=',')
standard_values = data['value'].values.reshape(-1, 1)

# 创建并拟合标准化器
scaler = StandardScaler().fit(standard_values)
minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(scaler.transform(standard_values))

def one_hot_encode(sequence):
    """将8个氨基酸的序列编码为one-hot格式"""
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 
          'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 
          'E', 'D', 'K', 'R']
    
    # 检查序列长度
    if len(sequence) != 8:
        raise ValueError("Input sequence must be 8 amino acids long")
    
    encoding = []
    for aa in sequence:
        if aa == 'X':  # 处理未知氨基酸
            encoding += [0.05]*20
        else:
            encoding += [1 if aa == aa_class else 0 for aa_class in AA]
    return np.array(encoding).reshape(8, 20)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预测氨基酸序列活性分数')
    parser.add_argument('input_file', help='输入序列文件路径')
    parser.add_argument('output_file', help='输出结果文件路径')
    parser.add_argument('--batch_size', type=int, default=1024, help='批处理大小')
    return parser.parse_args()

def predict_in_batches(model, input_file, output_file, batch_size):
    """批量预测并保存结果"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        batch = []
        for line in infile:
            sequence = line.strip()
            if sequence:
                batch.append(sequence)
                if len(batch) >= batch_size:
                    process_batch(model, batch, outfile)
                    batch = []
        
        if batch:  # 处理最后不足一个批次的数据
            process_batch(model, batch, outfile)

def process_batch(model, batch, outfile):
    """处理单个批次"""
    try:
        # 批量编码
        encoded_seqs = [one_hot_encode(seq) for seq in batch]
        
        # 转换为tensor并添加batch维度
        input_tensors = [tf.expand_dims(tf.constant(seq, dtype=tf.float32), axis=0) for seq in encoded_seqs]
        input_data = tf.concat(input_tensors, axis=0)
        
        # 批量预测
        normalized_scores = model.predict(input_data, batch_size=len(batch))
        
        # 逆标准化处理
        for seq, norm_score in zip(batch, normalized_scores[:,0]):
            score = minmax_scaler.inverse_transform([[norm_score]])
            original_score = scaler.inverse_transform(score)
            outfile.write(f"{seq}\t{round(original_score[0][0], 4)}\n")
    except Exception as e:
        print(f"处理批次时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    args = parse_args()
    model = tf.keras.models.load_model(model_save_path)
    predict_in_batches(model, args.input_file, args.output_file, args.batch_size)
