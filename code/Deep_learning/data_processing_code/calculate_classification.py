# 读取文件内容
file_path = 'stat.ref.denoise.txt'
output_path = 'stat.ref.denoise.classification1.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化R2和R4的总和
sum_R2 = 0
sum_R4 = 0

# 提取每行的R2和R4值并累加
for line in lines:
    parts = line.split()
    if len(parts) >= 7:  # 确保行有足够的数据
        R2 = int(parts[3])
        R4 = int(parts[5])
        sum_R2 += R2
        sum_R4 += R4

# 计算比值和差值，生成最终表格
results = []
for line in lines:
    parts = line.split()
    if len(parts) >= 7:
        sequence = parts[0]
        R2 = int(parts[3])
        R4 = int(parts[5])
        ratio_R2 = R2 / sum_R2
        ratio_R4 = R4 / sum_R4
        if ratio_R2 != 0 or ratio_R4 != 0:  # 排除R2和R4比值都为0的情况
            diff = ratio_R4 - ratio_R2
            flag = 1 if diff > 0 else 0
            results.append([sequence, f"{ratio_R2:.6f}", f"{ratio_R4:.6f}", f"{diff:.6f}", flag])

# 写入结果到文件
with open(output_path, 'w') as file:
    # 写入行头
    file.write("Sequence\tRatio_R2\tRatio_R4\tDiff_R4_R2\tFlag\n")
    # 写入数据
    for result in results:
        file.write('\t'.join(map(str, result)) + '\n')
