from keras.layers import LSTM,GRU,Dense,Dropout,Activation,Embedding,Flatten,Bidirectional,MaxPooling2D,Conv1D,SpatialDropout1D,MaxPooling1D
from tensorflow.keras.layers import Input, Dense, Multiply, Add, Conv1D, Concatenate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

maxlen =89
trunc_type='post'

oov_tok = "<OOV>"
vocab_size = 12

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
def positional_encoding(maxlen, num_hid):
        depth = num_hid/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1)
        return pos_encoding

def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = Conv1D(filters = filters,
               kernel_size = 1,
               padding = 'same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same', 
                          activation = 'tanh', 
                          dilation_rate = dilation_rate)(x)
        sigm_out = Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same',
                          activation = 'sigmoid', 
                          dilation_rate = dilation_rate)(x)
        x = Multiply()([tanh_out, sigm_out])
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = Add()([res_x, x])
    return res_x


feat_dim = 2
embed_dim = 2
num_heads = 2
ff_dim = 2
dropout_rate = 0.0
num_blocks = 2
def build_model_1():
    # INPUT
    inp = tf.keras.Input(shape=(89,))  # Input shape set to (89,)

    # Reshape to (89, 1) to make it compatible with Conv1D
    x = layers.Reshape((89, 1))(inp)

    # POSITIONAL ENCODING
    x = layers.Dense(feat_dim)(x)
    p = positional_encoding(89, feat_dim)
    x = x + p

    # FOUR of (2xCCN + 1xTRANSFORMER BLOCKS)
    for k in range(num_blocks):
        skip = x
        # Adjust wave_block to output 2D features
        x = wave_block(x, feat_dim, 3, 12)
        x = wave_block(x, feat_dim, 3, 12)
        x = TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)(x)
        x = 0.9 * x + 0.1 * skip

    # RNN BLOCK
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=128, return_sequences=True))(x)

    # OUTPUT
    x1 = tf.keras.layers.Dense(1, activation='sigmoid')(x)
      # Flatten the output to make it 2D

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x1)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy()  # Binary cross-entropy loss
    model.compile(loss=loss, optimizer=opt,metrics=['accuracy'])

    return model

model = build_model_1()
