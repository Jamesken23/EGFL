from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, ReLU, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Layer
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
import os

"""
TextCNN + Transformer network
ref: https://github.com/percent4/Keras_Transformer_Text_Classification/blob/master/src/model.py
"""


class MultiHeadAttention(Layer):
    """
    # Input
        three 3D tensor: Q, K, V
        each with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input0 steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = MultiHeadAttention(8,16)([embeddings,embeddings,embeddings]) # self Attention
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, key_size=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['key_size'] = self.key_size
        return config

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(units=self.out_dim, use_bias=False)

    def call(self, inputs):
        Q_seq, K_seq, V_seq = inputs

        Q_seq = self.q_dense(Q_seq)
        K_seq = self.k_dense(K_seq)
        V_seq = self.v_dense(V_seq)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.heads, self.key_size))
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.heads, self.key_size))
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.heads, self.size_per_head))

        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # Attention
        A = tf.einsum('bjhd,bkhd->bhjk', Q_seq, K_seq) / self.key_size ** 0.5
        A = K.softmax(A)

        O_seq = tf.einsum('bhjk,bkhd->bjhd', A, V_seq)
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.out_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class LayerNormalization(Layer):

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config['center'] = self.center
        config['scale'] = self.scale
        config['epsilon'] = self.epsilon
        return config

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

    def call(self, inputs):
        if self.center:
            beta = self.beta
        if self.scale:
            gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TransformerBlock(Layer):
    """
    # Input
        3D tensor: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = TransformerBlock(8,16,128)(embeddings)
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.ff_dim = ff_dim
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['ff_dim'] = self.ff_dim
        config['rate'] = self.rate
        return config

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)
        assert input_shape[-1] == self.heads * self.size_per_head
        self.att = MultiHeadAttention(heads=self.heads, size_per_head=self.size_per_head)
        self.ffn = Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.heads * self.size_per_head),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, inputs):
        attn_output = self.att([inputs, inputs, inputs])
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    
class CFG_TextCNN_Transformer:
    def __init__(self, opcode_data, graph_data, args, log_set):
        input1 = tf.keras.Input(shape=(args.SEQ_LEN), name='input1')
        # 额外添加了图数据 256维
        input2 = tf.keras.Input(shape=(args.EMBEDDING_LAYER_DIM), name='input2')
        
        (X_train, X_val, Y_train, Y_val) = opcode_data
        self.x_train, self.y_train = X_train, Y_train
        self.x_test, self.y_test = X_val, Y_val
        
        # ----------------------------------------添加了graph data------------------------------------------------------
        (graph_X_train, graph_X_val, graph_Y_train, graph_Y_val) = graph_data
        self.graph_x_train, self.graph_y_train = graph_X_train, graph_Y_train
        self.graph_x_test, self.graph_y_test = graph_X_val, graph_Y_val
        
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.weights_dir = args.weights_dir
        self.log_set = log_set
        self.log_name = args.log_name
        
        adam = Adam(learning_rate=args.lr)
        
        x_embedding = Embedding(args.MAX_WORDS, args.EMBEDDING_LAYER_DIM, input_length=self.x_train.shape[1])(input1)
        # add convolutional layer
        x_conv1 = Conv1D(filters=args.EMBEDDING_LAYER_DIM, kernel_size=5, activation='relu', padding="same")(x_embedding)
        x_conv1 = MaxPooling1D(pool_size=2)(x_conv1)
        # add transformer block
        x_trans = TransformerBlock(8, 16*2, args.EMBEDDING_LAYER_DIM)(x_conv1)
        x_trans = GlobalAveragePooling1D()(x_trans)
    
        # 对于graph数据，也加一层
        graph2vec = tf.keras.layers.Dense(args.EMBEDDING_LAYER_DIM, activation='relu')(input2)
        graphweight = tf.keras.layers.Dense(1, activation='sigmoid')(graph2vec)
        newgraphvec = tf.keras.layers.Multiply()([graph2vec, graphweight])
        
        mergevec = tf.keras.layers.Concatenate(axis=1)([x_trans, newgraphvec])
        
        x_final = tf.keras.layers.Dense(100, activation='relu')(mergevec)
        x_final = Dropout(args.dropout)(x_final)
        
        prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x_final)
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=[prediction])
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.summary()
        self.log_set.info(model.summary())

        self.model = model
        
    """
    Trains model
    """

    def train(self):
        # 如果没有权重文件夹，则立即创建
        weights_name = self.log_name.split(".")[0]
        self.weights_dir = os.path.join(self.weights_dir, weights_name)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
            
        weights_path = os.path.join(self.weights_dir, weights_name+"-{epoch:02d}-{val_acc:.3f}.pkl")
        
        checkpoint = ModelCheckpoint(weights_path, 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max',
                             period=1)

        callbacks_list = [checkpoint]

        # 组合self.x_train, self.graph_x_train两种训练数据
        multi_train_data = [self.x_train, self.graph_x_train]
        # 组合self.x_train, self.graph_x_train两种训练数据
        multi_valid_data = [self.x_test, self.graph_x_test]
        
        history = self.model.fit(multi_train_data, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(multi_valid_data, self.y_test), callbacks=callbacks_list)

        print("Training epoch is {0}, loss is {1}, accuracy is {2}".format(history.epoch, history.history['loss'], history.history['acc']))
        self.log_set.info("Training epoch is {0}, loss is {1}, accuracy is {2}, Test loss is {3}, Test acc is {4}".format(history.epoch, history.history['loss'], history.history['acc'],  history.history['val_loss'], history.history['val_acc']))
        # 保存最大的测试精确度
        val_acc_list  = list(history.history['val_acc'])
        max_test_acc = max(val_acc_list)
        max_acc_epoch = val_acc_list.index(max_test_acc) + 1
        print("Epoch {0}, we cam get max test accuracy {1}".format(max_acc_epoch, max_test_acc))
        self.log_set.info("Epoch {0}, we cam get max test accuracy {1}".format(max_acc_epoch, max_test_acc))
        self.best_model = None
        # 保存最终的模型，用于记录全局训练损失
        final_model_path = os.path.join(self.weights_dir, weights_name+".h5")
        self.model.save_weights(final_model_path)
        if 0 <= max_acc_epoch < 10:
            self.best_epoch = "-0"+str(max_acc_epoch)+"-"
        else:
            self.best_epoch = "-"+str(max_acc_epoch)+"-"


    """
    Tests accuracy of model
    Loads weights from file if no weights are attached to model object
    """

    def test(self):
        # print(self.best_epoch, self.best_model)
        for i in os.listdir(self.weights_dir):
            if self.best_epoch in i:
                self.best_model = os.path.join(self.weights_dir, i)
        if self.best_model is not None:
            self.model.load_weights(self.best_model)
            print("we have loaded best model {0}".format(self.best_model))
            self.log_set.info("we have loaded best model {0}".format(self.best_model))
        
        # 组合self.x_train, self.graph_x_train两种训练数据
        multi_valid_data = [self.x_test, self.graph_x_test]
        
        values = self.model.evaluate(multi_valid_data, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        predictions = (self.model.predict(multi_valid_data, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions, axis=1).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
        self.log_set.info("Accuracy: {0}, FPR: {1}, FN: {2}, Recall(TPR): {3}, Precision: {4}, F1 score: {5}".format((tp + tn) / (tp + tn + fp + fn), fp / (fp + tn), fn / (fn + tp), recall, precision, (2 * precision * recall) / (precision + recall)))
        