import numpy as np
import tensorflow as tf
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.99, help='概率阈值')
    parser.add_argument('--batch_size', type=int, default=1024, help='批处理大小')
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    return tf.keras.models.load_model(checkpoint_path)

def preprocess_input(sequence):
    """预处理输入序列"""
    AA = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    max_length = 8
    processed_seq = sequence.ljust(max_length, 'I')[:max_length]
    encoding = []
    for aa in processed_seq:
        code = [1 if aa == aa_type else 0 for aa_type in AA]
        encoding.append(code)
    return np.array(encoding, dtype=np.float32).reshape(1, max_length, 20)

def predict_in_batches(model, input_file, output_file, threshold, batch_size):
    """分批处理预测任务"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        batch = []
        for line in infile:
            sequence = line.strip()
            if sequence:
                batch.append(sequence)
                if len(batch) >= batch_size:
                    # 处理当前批次
                    input_data = np.vstack([preprocess_input(seq) for seq in batch])
                    batch_pred = model.predict(input_data, batch_size=len(batch))
                    for seq, prob in zip(batch, batch_pred[:,0]):
                        if prob >= threshold:
                            outfile.write(f"{seq}\t{prob:.4f}\n")
                    batch = []  # 清空批次
        
        # 处理最后不足一个批次的数据
        if batch:
            input_data = np.vstack([preprocess_input(seq) for seq in batch])
            batch_pred = model.predict(input_data, batch_size=len(batch))
            for seq, prob in zip(batch, batch_pred[:,0]):
                if prob >= threshold:
                    outfile.write(f"{seq}\t{prob:.4f}\n")

if __name__ == "__main__":
    args = parse_args()
    model = load_checkpoint("../model/CNN_classification")
    predict_in_batches(model, args.input_file, args.output_file, args.threshold, args.batch_size)
