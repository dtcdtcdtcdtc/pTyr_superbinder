import pandas as pd
import numpy as np

# 读取数据
file_path = 'stat.ref.denoise.10.regression.txt'
data = pd.read_csv(file_path, sep='\t')

# 对Diff_R4_R2列进行排序
data_sorted = data.sort_values(by='Diff_R4_R2', ascending=False)

# 选择Diff_R4_R2为正的行作为正样本
positive_samples = data_sorted[data_sorted['Diff_R4_R2'] > 0]

# 选择Diff_R4_R2为负的行作为负样本
negative_samples = data_sorted[data_sorted['Diff_R4_R2'] < 0]

# 计算正样本的数量
num_positive_samples = len(positive_samples)

# 使用numpy.linspace生成均匀间隔的索引
indices = np.linspace(0, len(negative_samples) - 1, num=num_positive_samples, dtype=int)
selected_negative_samples = negative_samples.iloc[indices]

# 选择需要的列并重命名
positive_samples = positive_samples[['Sequence', 'Diff_R4_R2']].rename(columns={'Sequence': 'sequence', 'Diff_R4_R2': 'value'})
selected_negative_samples = selected_negative_samples[['Sequence', 'Diff_R4_R2']].rename(columns={'Sequence': 'sequence', 'Diff_R4_R2': 'value'})

# 输出正样本和选定的负样本的数量
print(f'Number of positive samples: {num_positive_samples}')
print(f'Selected negative samples: {selected_negative_samples["sequence"].tolist()}')

# 输出到文件
positive_samples.to_csv('positive_samples.txt', index=False, sep='\t')
selected_negative_samples.to_csv('negative_samples.txt', index=False, sep='\t')
