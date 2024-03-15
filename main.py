import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
# Checking eager execution 
# tf.config.run_functions_eagerly(True)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from logs.log_helper import get_logger, get_log_path
from Networks.network import Feature_fusion_module
from parser_set import parameter_parser

import os, json, time
import numpy as np


args = parameter_parser()


# obtain the original data and label
def get_opcode_and_label(original_dataset_path="Dataset/reentrancy/opcodes/"):
    op_data, graph_data, op_label = [], [], []
    
    for i in os.listdir(original_dataset_path):
        if ".txt" not in i:
            continue
        op_path = os.path.join(original_dataset_path, i)
        with open(op_path, 'r') as f:
            try:
                js = f.read()
                op_dic = json.loads(js)
            except:
                continue
        if op_dic['simplified_opcode'] == "":
             continue 
        op_data.append(op_dic['simplified_opcode'])
        graph_data.append(op_dic['cfg_graph'])
        op_label.append(int(i.split('_')[-1].split('.')[0]))
    print("The length of train data and label is {0} and {1}".format(len(op_data), len(op_label)))
    return np.array(op_data), np.array(graph_data), np.array(op_label)


# Get the train set and valid set
def get_token_data(MAX_WORDS, SEQ_LEN, op_data, log_set):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
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


def get_weights_logs_path(model_name, vul_type):
    first_weights_dir = os.path.join(args.weights_dir, vul_type)
    if not os.path.exists(first_weights_dir):
        os.mkdir(first_weights_dir)
    second_weights_dir = os.path.join(first_weights_dir, model_name)
    if not os.path.exists(second_weights_dir):
        os.mkdir(second_weights_dir)
    args.weights_dir = second_weights_dir
    
    
    first_logs_dir = os.path.join(args.log_dir, vul_type)
    if not os.path.exists(first_logs_dir):
        os.mkdir(first_logs_dir)
    second_logs_dir = os.path.join(first_logs_dir, model_name)
    if not os.path.exists(second_logs_dir):
        os.mkdir(second_logs_dir)
    args.log_dir = second_logs_dir


if __name__ == "__main__":
    args.weights_dir = '/Weights'
    args.log_dir = '/logs'
    
    is_balance = True
    args.model_name = 'EGFL'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.sc_vul_type = 'block number dependency'
    args.train_dataset = '/Dataset/BN/Train_data'
    args.val_dataset = '/Dataset/BN/Valid_data'
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # args.sc_vul_type = 'unchecked external call'
    # args.train_dataset = '/Dataset/UC/Train_data'
    # args.val_dataset = '/Dataset/UC/Valid_data'
    
    op_X_train, graph_X_train, Y_train = get_opcode_and_label(args.train_dataset)
    op_X_valid, graph_X_val, Y_val = get_opcode_and_label(args.val_dataset)
    
    all_train_data = []
    all_train_data.extend(op_X_train.tolist())
    all_train_data.extend(op_X_valid.tolist())
    all_train_data = np.array(all_train_data)
    
    get_weights_logs_path(args.model_name, args.sc_vul_type)

    log_path, log_name = get_log_path(args)
    log_set = get_logger(log_path)
    log_set.info("Our Model's Args is: ", args)
    
    args.log_name = log_name
    
    all_token_train_data = get_token_data(args.MAX_WORDS, args.SEQ_LEN, all_train_data, log_set)
    X_train, X_val = all_token_train_data[:len(op_X_train)], all_token_train_data[len(op_X_train):]
    
    
    log_set.info("Final X_train shape is {0}, X_val shape is {1}, graph_X_train shape is {2}, graph_X_val shape is {3}".format(X_train.shape, X_val.shape, graph_X_train.shape, graph_X_val.shape))
    

    model = Feature_fusion_module((X_train, X_val, Y_train, Y_val), (graph_X_train, graph_X_val, Y_train, Y_val), args, log_set)  
        
    
    # Start training
    start_time = time.time()
    
    log_set.info("start {0} training!".format(args.model_name))
    model.train()
    print("End {0} training!".format(args.model_name))
    log_set.info("End {0} training!".format(args.model_name))
    
    # Start testing
    log_set.info("start {0} testing!".format(args.model_name))
    model.test()
    
    end_time = time.time()
    avg_time = (end_time - start_time)/args.epochs
    print("End {0} testing! average time is {1}".format(args.model_name, avg_time))
    log_set.info("End {0} testing!".format(args.model_name, avg_time))