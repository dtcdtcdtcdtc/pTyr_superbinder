#gpu内存
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from tensorflow.keras import layers, Sequential, losses, optimizers
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score
from tensorflow.keras.constraints import max_norm
from tensorflow import  keras
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from bioencoder.encoder import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler,scale


physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)  # 注意 ，这里为tensorflow2.0版本，与第1.0有差距。

standard_file_path=""#用于数据标准化处理的文件路径。一般该文件须包括用于模型训练和测试的所有数据。
model_save_path=""#模型保存路径
data_path=""#训练数据集以及独立测试集的保存路径
result_save_path=""#预测结果的保存路径

StandardScalerIN=pd.read_csv(standard_file_path,sep='\t')
StandardScaler_range=StandardScalerIN['value']
StandardScaler_range=StandardScaler_range.values.reshape(-1, 1)
mm = StandardScaler()
mm=mm.fit(StandardScaler_range)
value_transform=mm.transform(StandardScaler_range)

mm_minmax = MinMaxScaler(feature_range=(-1, 1))
mm_minmax=mm_minmax.fit(value_transform)
def process(inputfile_path):
    data = pd.read_csv(inputfile_path, sep='\t', header=0)
    label = data['value']
    inputfile = data['sequence']
    sequence_list = []
    line_number = 0
    for line in inputfile:
        line_number = line_number + 1

    label_reshape = label.values.reshape(-1, 1)

    label_MinMaxScaler = mm.transform(label_reshape)
    label_MinMaxScaler=mm_minmax.transform(label_MinMaxScaler)


    return inputfile, line_number, label,label_MinMaxScaler
'''对完成分词操作的列表进行one-hot编码 ↓ '''
def One_Hot(sequence,line_number):
    AA=['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']
    AA2=['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    encodings = []
    for seq_line in sequence:
        code = []
        seq_line=list(seq_line)
        for aa in seq_line:
            if aa == 'X':
                for aa1 in AA2:
                    tag = 0.05 if aa == aa1 else 0
                    code.append(tag)
            else:
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
        encodings.append(code)
    np.array(encodings)
    encodings=np.reshape(encodings,(line_number,8,20))
    return encodings
'''搭建网络结构，并且进行编译 ↓ '''

def build_network():
    # 先创建包含多网络层的列表
    conv_layers = [

        layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu,input_shape=( 8, 20)),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=9, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(2, data_format="channels_first"),
        layers.Dropout(0.5),
        layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu),
        layers.MaxPooling1D(pool_size=2, strides=1),
        layers.Dropout(0.5),
    ]

    fc_layers = [
        layers.Dense(64, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(32, activation=tf.nn.relu),
        layers.MaxPooling1D(2),
        layers.Dense(8, activation=tf.nn.relu),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation=tf.nn.tanh)
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.build(input_shape=[None, 8, 20])
    network.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])  # optimizers.RMSprop()，此时的loss函数为均方差，用于回归预测，决定系数R2（coefficient ofdetermination）常常在线性回归中被用来表征有多少百分比的因变量波动被回归线描述。如果R2 =1则表示模型完美地预测了目标变量。(lr=0.01,rho=0.9, epsilon=1e-06)
    network.summary()
    return network


'''将数据类型进行转换 ↓ '''
def coeff_determination(y_true, y_pred):
    SS_res =K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return x, y

'''下面的evaluate自定义函数的作用是对训练完成的模型进行评估 ↓ '''
def evaluate(X, Y,Y_train_noNormalization,X_vali, Y_vali,Y_vali_noNormalization, X_TEST, Y_TEST,Y_test_noNormalization, batch_size=128, epochs=100,line_number_train=0,line_number_vali=0,line_number_test=0):
    classes = sorted([0, 1])
    print(Y)
    X_train, y_train = X, Y
    '''将相应数据集转换为one-hot编码形式'''

    ############One_Hot#####################
    X_train = One_Hot(X_train, line_number_train)  # 训练集one-hot编码之后的得到的数据集
    X_vali = One_Hot(X_vali, line_number_vali)  # 验证集one-hot编码之后的得到的数据集
    X_test = One_Hot(X_TEST, line_number_test)  # 测试集one-hot编码之后的得到的数据集
    ############One_Hot#####################


    X_train_t = X_train
    X_vali_t = X_vali
    X_test_t = X_test

    X_test_t = tf.cast(X_test_t, dtype=tf.float32)

    '''下面六行命令之间两两类似，都是先将X与Y通过生成tf.data的形式绑定在一块，随后在下一行命令里面进行数据类型的变换，以及批量化处理 ↓ '''
    # 构建训练集对象，随机打乱，预处理，批量化
    train_db = tf.data.Dataset.from_tensor_slices((X_train_t,y_train))  # 首先将pandas dataframe 数据格式转变为 tf.data 格式的数据集形式，这样是为了下一步数据进行打乱处理做准备，转换为tf.data格式之后，序列与label值则昂订到一块，我们就可以对其进行同步处理
    train_db = train_db.shuffle(len(X)).map(preprocess).batch(batch_size)  # 其中map函数是用来将序列做一键预处理用的。该行命令：先对数据进行随机化处理，随后使用map函数进行预处理，使用batch函数指定批次数量
    # 构建验证集对象，预处理，批量化
    vali_db = tf.data.Dataset.from_tensor_slices((X_vali_t, Y_vali))
    vali_db = vali_db.shuffle(len(X_vali_t)).map(preprocess).batch(batch_size)
    # 构建测试集对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((X_test_t, Y_TEST))
    test_db = test_db.shuffle(len(X_test_t)).map(preprocess).batch(batch_size)
    '''调用已经搭建好的神经网络，并进行训练，并将独立测试集的测试结果输出'''
    network = build_network()
    model_save_file_path=model_save_path
    checkpoint = ModelCheckpoint(model_save_file_path, monitor='val_mae', verbose=1, save_best_only=True, mode='auto',period=50, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [early_stopping, checkpoint]

    history = network.fit(train_db, epochs=epochs, validation_data=vali_db,  verbose=1,callbacks = [callbacks_list])  # ,callbacks = [early_stopping]
    print("Independent test:", network.evaluate(test_db))
    tmp_result = np.zeros((len(Y_test_noNormalization), len(classes)))
    predict = network.predict(X_test_t, batch_size=batch_size)
    print(predict)

    predict=mm_minmax.inverse_transform(predict)
    predict = mm.inverse_transform(predict)
    print()
    R_squire = r2_score(Y_test_noNormalization, predict)
    tmp_result[:, 0], tmp_result[:, 1] = Y_test_noNormalization, predict[:, 0]
    network.save(model_save_file_path, save_format='tf')
    return tmp_result, history, R_squire

'''下面的main自定义函数的作用主要为对以上的函数进行调用（主函数）'''

#储存结果的自定义函数
def save_predict_result(data, output):
    with open(output, 'w') as f:
        f.write('value'+'\t'+'predict'+'\n')
        for i in data:
            f.write('%f\t%f\n' % (i[0], float(i[1])))
    return None


#运行main函数
def random_dataset_number(data,label,sample_num):
    '''随机产生一组数，加上random.seed(1)之后就可以保持每次循环所用的数是一样的，这里选择的是直接输入样本数量；还有另外一种方法，是输入取值所占百分比:1、平衡正负样本'''
    import random
    random.seed(1)  # 此行代码：括号里面是0，则每次产生的数组不一样，是1的话每次产生的数组是一样的
    sample_list = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list, sample_num)  # 随机选取出了 [3, 4, 2, 0]
    sample_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]

    '''保存list'''
    name1 = ['sequence']
    sequence = pd.DataFrame(columns=name1, data=sample_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name2 = ['label']
    label = pd.DataFrame(columns=name2, data=sample_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    return sample_data,sample_label,sequence,label
def random_dataset_percent(data,label,percent):
    '''随机产生一组数，加上random.seed(1)之后就可以保持每次循环所用的数是一样的，这里选择的是独立测试集所占百分比:2、提取独立测试集的自定函数'''

    sample_num = int(percent * len(data))  # 假设取20%的数据作为独立测试集
    random.seed(2)  # 此行代码：括号里面是0，则每次产生的数组不一样，是非0的话每次产生的数组是一样的，保证每次提取的独立测试集都相同
    sample_list_all = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list_all, sample_num)  # 随机选取出了 [3, 4, 2, 0]
    sample_difference = list(set(sample_list_all).difference(set(sample_list)))
    sample_test_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    sample_test_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]
    sample_train_data = [data[i] for i in sample_difference]  # ['d', 'e', 'c', 'a']
    sample_train_label = [label[i] for i in sample_difference]  # [3, 4, 2, 0]
    '''保存list'''
    name = ['sequence']
    sequence = pd.DataFrame(columns=name, data=sample_test_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name = ['label']
    label = pd.DataFrame(columns=name, data=sample_test_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    '''保存train_vali的list'''
    name = ['sequence']
    sequence_train = pd.DataFrame(columns=name, data=sample_train_data)  # 定义数据的表头以及数据,此行命令保存的是数据那一列
    name = ['label']
    label_train = pd.DataFrame(columns=name, data=sample_train_label)  # 定义数据的表头以及数据,此行命令保存的是label那一列
    return sample_test_data,sample_test_label,sample_train_data,sample_train_label,sequence,label,sequence_train,label_train
'''使用bioinformatics的数据进行处理↓'''
def main():
    os.chdir(data_path)
    epoch = 1000
    X_train,line_number_train,Y_train,Y_train_MinMaxScaler=process("train")
    X_TEST, line_number_test,Y_TEST,Y_TEST_MinMaxScaler=process("test")
    ###########"""数据经过标准化之后，再次经过归一化处理到（-1,1）。"""#########################


    x_vali, y_vali = X_TEST, Y_TEST_MinMaxScaler
    line_number_vali=line_number_test
    os.chdir(result_save_path)
    ind_res_test, history,R_squire_test = evaluate(X_train, Y_train_MinMaxScaler,Y_train,x_vali, y_vali,y_vali, X_TEST,Y_TEST_MinMaxScaler,Y_TEST,epochs=epoch, batch_size=128,line_number_train=line_number_train,line_number_vali=line_number_vali,line_number_test=line_number_test) #,R_squire_valiind_res_vali,
    save_predict_result(ind_res_test, 'regression_predict.txt')
    acc = history.history['mae']
    val_acc = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print('R_squire_test:' + str(R_squire_test))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training mae')
    plt.plot(val_acc, label='Validation mae')
    plt.title('Training and Validation mae')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('regression_predict.pdf')

if __name__ == '__main__':
    main()
'''1、平衡正负样本
   2、先把独立测试集按比例拆分出来（正负样本分别提取、然后合并），加上判断测试集文件是否存在的命令
   3、取出独立测试集之后的正样本负样本都随机按比例拆分成训练集、验证集，训练过程中保证每次所拆分的训练集与验证集保持不变
   4、按照原来代码中就有的函数将label与sequence合成一个文件（确保一一对应），然后随机化打乱处理
   5、训练集：验证集：测试集=6:2:2'''
