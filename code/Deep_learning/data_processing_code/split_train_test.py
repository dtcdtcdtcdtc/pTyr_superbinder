def split_data(input_file, train_file, test_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # 保留表头
    header = lines[0]
    
    # 初始化训练集和测试集，包括表头
    train_data = [header]
    test_data = [header]
    
    # 遍历除表头外的数据行
    for index in range(1, len(lines)):
        if (index - 1) % 5 == 0:
            test_data.append(lines[index])
        else:
            train_data.append(lines[index])
    
    # 写入训练集数据
    with open(train_file, 'w') as file:
        file.writelines(train_data)
    
    # 写入测试集数据
    with open(test_file, 'w') as file:
        file.writelines(test_data)

# 调用函数
split_data('out.txt', 'train_set.txt', 'test_set.txt')
