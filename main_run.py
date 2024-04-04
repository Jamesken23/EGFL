import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
# Checking eager execution 
# tf.config.run_functions_eagerly(True)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from logs.log_helper import get_logger, get_log_path
from Networks.egfl_net import egfl_model
from parser_set import parameter_parser

import os, json, time
import numpy as np


args = parameter_parser()


# 获取原始数据集和标签
def get_data_label(original_dataset_path):
    # opcode_act
    # 定义原始数据集和标签的列表
    op_data, graph_data, op_label = [], [], []
    for i in os.listdir(original_dataset_path):
        # 遍历读取数据
        if ".txt" not in i:
            continue
        op_path = os.path.join(original_dataset_path, i)
        with open(op_path, 'r') as f:
            try:
                js = f.read()
                op_dic = json.loads(js)
            except:
                continue
        if op_dic['cfg_bfs_act'] == "":
             continue 
        op_data.append(op_dic['cfg_bfs_act'])
        graph_data.append(op_dic['cfg_graph'])
        op_label.append(int(i.split('_')[-1].split('.')[0]))
    print("The length of train data and label is {0} and {1}".format(len(op_data), len(op_label)))
    # print("op_X_data: {0}, graph_X_data: {1}".format(op_data[0], graph_data[0]))
    return np.array(op_data), np.array(graph_data), np.array(op_label)


# 将原始数据集变成词向量，并拆分成训练集和测试集
def get_token_data(MAX_WORDS, SEQ_LEN, op_data, log_set):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    # in disassembled data 根据text创建一个词汇表。其顺序依照词汇在文本中出现的频率。
    tokenizer.fit_on_texts(op_data) 
    
    # --------------------- set MAX_WORDS --------------------------------------
    word_index = tokenizer.word_index
    log_set.info('Found %s unique tokens.' % len(word_index))

    # tokenizing sequences
    # --------------------- set SEQ_LEN --------------------------------------
    X = tokenizer.texts_to_sequences(op_data)
    X = pad_sequences(X, maxlen=args.SEQ_LEN)
    log_set.info('Shape of data tensor: {0}, max length of sentence token is {1}'.format(X.shape, args.SEQ_LEN))

    return X


# 使用类不平衡方法平衡数据集
# --------------------------------------扩充图数据----------------------------------------------------------
def get_balanced_train_test_data(X_train, X_val, Y_train, Y_val, graph_X_train, graph_X_val, log_set):
    # --------------------------------------控制随机状态，可以保证每次生成的数一样--------------------------------------------------------
    sm = SMOTE(random_state=39, k_neighbors=3)
    X_train, Y_train_copy = sm.fit_resample(X_train, Y_train)
    graph_X_train, graph_Y_train = sm.fit_resample(graph_X_train, Y_train)
    
    log_set.info("Y_train_copy is graph_Y_train : {0}".format((Y_train_copy==graph_Y_train).all()))
    
    X_val, Y_val_copy = sm.fit_resample(X_val, Y_val)
    graph_X_val, graph_Y_val = sm.fit_resample(graph_X_val, Y_val)
    
    log_set.info("Y_val_copy is graph_Y_val : {0}".format((Y_val_copy==graph_Y_val).all()))
    
    log_set.info("SMOTE X_train shape is {0}, X_val shape is {1}, Y_train shape is {2}, Y_val shape is {3}".format(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape))
    
    # 添加个控制状态，如果扩充后的操作码数据和CFG图数据没有对齐，则直接舍弃退出
    if (Y_train_copy==graph_Y_train).all() != True:
        print("Your X_train and graph_X_train have not the same dimension")
        print("Y_train_copy:", Y_train_copy)
        print("graph_Y_train:", graph_Y_train)
        exit()
    if (Y_val_copy==graph_Y_val).all() != True:
        print("Your X_val and graph_X_val have not the same dimension")
        print("Y_val_copy:", Y_val_copy)
        print("graph_Y_val:", graph_Y_val)
        exit()
    
    return X_train, X_val, Y_train_copy, Y_val_copy, graph_X_train, graph_X_val


# 根据漏洞类型自动生成相应的权重和日志地址
def get_weights_logs_path(model_name, vul_type):
    # 按照"Weights/Vul/Network/"的顺序依次生成一二级权重保存文件夹
    first_weights_dir = os.path.join(args.weights_dir, vul_type)
    if not os.path.exists(first_weights_dir):
        os.mkdir(first_weights_dir)
    second_weights_dir = os.path.join(first_weights_dir, model_name)
    if not os.path.exists(second_weights_dir):
        os.mkdir(second_weights_dir)
    args.weights_dir = second_weights_dir
    
    # 按照"logs/Vul/Network/"的顺序依次生成一二级权重保存文件夹
    first_logs_dir = os.path.join(args.log_dir, vul_type)
    if not os.path.exists(first_logs_dir):
        os.mkdir(first_logs_dir)
    second_logs_dir = os.path.join(first_logs_dir, model_name)
    if not os.path.exists(second_logs_dir):
        os.mkdir(second_logs_dir)
    args.log_dir = second_logs_dir


if __name__ == "__main__":
    # 原始代码及相应标签存放的文件位置
    args.weights_dir = '/Weights_CFG_SimOp'
    args.log_dir = '/logs_CFG_SimOp'

    # 是否balance训练数据
    is_balance = False
    args.model_name = 'egfl_model'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.sc_vul_type = 'block number dependency'
    args.train_dataset = '/Dataset/BN/Train_data'
    args.val_dataset = '/Dataset/BN/Valid_data'
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # args.sc_vul_type = 'unchecked external call'
    # args.train_dataset = '/Dataset/UC/Train_data'
    # args.val_dataset = '/Dataset/UC/Valid_data'
    
    op_X_train, graph_X_train, Y_train = get_data_label(args.train_dataset)
    op_X_valid, graph_X_val, Y_val = get_data_label(args.val_dataset)
    
    all_train_data = []
    all_train_data.extend(op_X_train.tolist())
    all_train_data.extend(op_X_valid.tolist())
    all_train_data = np.array(all_train_data)
    
    # 根据漏洞类型自动生成相应的权重和日志地址
    get_weights_logs_path(args.model_name, args.sc_vul_type)

    # 开始打印日志信息
    log_path, log_name = get_log_path(args)
    log_set = get_logger(log_path)
    log_set.info("Our Model's Args is: ", args)
    
    args.log_name = log_name
    
    # 将原始数据集变成词向量，并拆分成训练集和测试集
    all_token_train_data = get_token_data(args.MAX_WORDS, args.SEQ_LEN, all_train_data, log_set)
    
    X_train, X_val = all_token_train_data[:len(op_X_train)], all_token_train_data[len(op_X_train):]
    
    # 使用类不平衡方法平衡数据集
    if is_balance == True:
        X_train, X_val, Y_train, Y_val, graph_X_train, graph_X_val = get_balanced_train_test_data(X_train, X_val, Y_train, Y_val, graph_X_train, graph_X_val, log_set)
    
    log_set.info("Final X_train shape is {0}, X_val shape is {1}, graph_X_train shape is {2}, graph_X_val shape is {3}".format(X_train.shape, X_val.shape, graph_X_train.shape, graph_X_val.shape))
    
    if args.model_name == 'egfl_model':
        model = egfl_model((X_train, X_val, Y_train, Y_val), (graph_X_train, graph_X_val, Y_train, Y_val), args, log_set)
        
    
    # 开始训练
    start_time = time.time()
    
    log_set.info("start {0} training!".format(args.model_name))
    model.train()
    print("End {0} training!".format(args.model_name))
    log_set.info("End {0} training!".format(args.model_name))
    
    # 开始测试
    log_set.info("start {0} testing!".format(args.model_name))
    model.test()
    
    end_time = time.time()
    avg_time = (end_time - start_time)/args.epochs
    print("End {0} testing! average time is {1}".format(args.model_name, avg_time))
    log_set.info("End {0} testing!".format(args.model_name, avg_time))