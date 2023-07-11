from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
# tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, ReLU, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Dropout, Layer, LayerNormalization, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from einops import rearrange
from einops.layers.tensorflow import Rearrange

import os, einops

"""
TextCNN + Transformer network
ref: https://github.com/percent4/Keras_Transformer_Text_Classification/blob/master/src/model.py
"""


class Attention(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512, **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = Dense(inner_dim, use_bias=False)
        self.to_kv = Dense(inner_dim * 2, use_bias=False)
        self.to_out = Dense(dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = Dropout(dropout)

    def call(self, inputs, context=None, mask=None, context_mask=None):
        n = inputs.shape[-2]
        heads = self.heads
        max_pos_emb = self.max_pos_emb
        if context is None:
            has_context = False
            context = inputs
        else:
            has_context = True

        kv = tf.split(self.to_kv(context), num_or_size_splits=2, axis=-1)
        q, k, v = (self.to_q(inputs), *kv)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), (q, k, v)
        )
        dots = tf.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        seq = tf.range(n)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = (
            tf.clip_by_value(
                dist, clip_value_min=-max_pos_emb, clip_value_max=max_pos_emb
            )
            + max_pos_emb
        )
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = tf.einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if mask is not None or context_mask is not None:
            if mask is not None:
                mask = tf.ones(*inputs.shape[:2])
            if not has_context:
                if context_mask is None:
                    context_mask = mask
            else:
                if context_mask is None:
                    context_mask = tf.ones(*context.shape[:2])
            mask_value = -tf.experimental.numpy.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots = tf.where(mask, mask_value, dots)

        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        out, gate = tf.split(inputs, 2, axis=self.dim)
        return out * tf.sigmoid(gate)


class DepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding, **kwargs):
        super(DepthwiseLayer, self).__init__(**kwargs)
        self.padding = padding
        self.chan_in = chan_in
        # self.conv = Conv1D(chan_out, 1)
        self.conv = Conv1D(chan_out, 1)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])
        padded = tf.zeros(
            [self.chan_in * self.chan_in] - tf.shape(inputs), dtype=inputs.dtype
        )
        inputs = tf.concat([inputs, padded], 0)
        inputs = tf.reshape(inputs, [-1, self.chan_in, self.chan_in])

        return self.conv(inputs)


class Scale(tf.keras.layers.Layer):
    def __init__(self, scale, fn, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = scale
        self.fn = fn

    def call(self, inputs, **kwargs):
        return self.fn(inputs, **kwargs) * self.scale


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fn = fn

    def call(self, inputs, **kwargs):
        inputs = self.norm(inputs)
        return self.fn(inputs, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, activation=Swish()),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim, input_dim=dim * mult),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, causal, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.causal = causal

    def call(self, inputs):
        if not self.causal:
            return tf.keras.layers.BatchNormalization(axis=-1)(inputs)
        return tf.identity(inputs)


class ConformerConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
        **kwargs
    ):
        super(ConformerConvModule, self).__init__(**kwargs)

        inner_dim = dim * expansion_factor
        if not causal:
            padding = (kernel_size // 2, kernel_size // 2 - (kernel_size + 1) % 2)
        else:
            padding = (kernel_size - 1, 0)

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(axis=-1),
                Rearrange("b n c -> b c n"),
                Conv1D(inner_dim * 2, 1),
                GLU(dim=1),
                # DepthwiseLayer(
                #     inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
                # ),
                BatchNorm(causal=causal),
                Swish(),
                Conv1D(dim, 1),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        **kwargs
    ):
        super(ConformerBlock, self).__init__(**kwargs)
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs, mask=None):
        inputs = self.ff1(inputs) + inputs
        inputs = self.attn(inputs, mask=mask) + inputs
        # inputs = self.conv(inputs) + inputs
        inputs = self.conv(inputs)
        inputs = self.ff2(inputs) + inputs
        inputs = self.post_norm(inputs)
        return inputs
    
    
class CFG_TextCNN_Conformer:
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
        
        # add contransformer block
        x_trans = ConformerBlock(args.EMBEDDING_LAYER_DIM)(x_conv1)
        x_trans = GlobalAveragePooling1D()(x_trans)
        
        # 对于graph数据，也加一层
        graph2vec = tf.keras.layers.Dense(args.EMBEDDING_LAYER_DIM, activation='relu')(input2)
        graphweight = tf.keras.layers.Dense(1, activation='sigmoid')(graph2vec)
        newgraphvec = tf.keras.layers.Multiply()([graph2vec, graphweight])
        
        trans2vec = tf.keras.layers.Dense(args.EMBEDDING_LAYER_DIM, activation='relu')(x_trans)
        transweight = tf.keras.layers.Dense(1, activation='sigmoid')(trans2vec)
        newtransvec = tf.keras.layers.Multiply()([trans2vec, transweight])
        
        mergevec = tf.keras.layers.Concatenate(axis=1)([newtransvec, newgraphvec])
        x_final = tf.keras.layers.Dense(100, activation='relu')(mergevec)
        x_final = Dropout(args.dropout)(x_final)
        
        prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x_final)
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=[prediction])
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.summary()

        self.log_set.info("our model structure: ", model.summary())

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

    """
    Tests accuracy of model
    Loads weights from file if no weights are attached to model object
    """

    def test(self):
        # 组合self.x_train, self.graph_x_train两种训练数据
        multi_valid_data = [self.x_test, self.graph_x_test]
        
        values = self.model.evaluate(multi_valid_data, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        predictions = (self.model.predict(multi_valid_data, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
        self.log_set.info("Accuracy: {0}, FPR: {1}, FN: {2}, Recall(TPR): {3}, Precision: {4}, F1 score: {5}".format((tp + tn) / (tp + tn + fp + fn), fp / (fp + tn), fn / (fn + tp), recall, precision, (2 * precision * recall) / (precision + recall)))
        