# 读取文件内容
file_path = 'stat.ref.denoise.10.txt'
output_path = 'stat.ref.denoise.10.regression.txt'

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
        #if R2 == 0 and R4 == 0:  # 如果R2和R4都为0，则跳过这行
        #    continue
        #if R2 == 0 or R4 == 0:  # 如果R2或R4为0
        #    R2 += 1
        #    R4 += 1
        R2_adjusted = R2 + 1 if R2 == 0 else R2  # 只有当R2为0时才加10
        R4_adjusted = R4 + 1 if R4 == 0 else R4  # 只有当R4为0时才加10
        ratio_R2 = R2_adjusted / sum_R2
        ratio_R4 = R4_adjusted / sum_R4
        if ratio_R2 != 0 or ratio_R4 != 0:  # 排除R2和R4比值都为0的情况
            import math
            diff = math.log10(ratio_R4 / ratio_R2)   # 使用log10计算比值，避免分母为0
            flag = 1 if diff > 0 else 0
            results.append([sequence, R2, R4, f"{ratio_R2:.6f}", f"{ratio_R4:.6f}", f"{diff:.6f}", flag])
# 写入结果到文件
with open(output_path, 'w') as file:
    # 写入行头
    file.write("Sequence\tR2\tR4\tRatio_R2\tRatio_R4\tRatio_R4_R2\tFlag\n")
    # 写入数据
    for result in results:
        file.write('\t'.join(map(str, result)) + '\n')
