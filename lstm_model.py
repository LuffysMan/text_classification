'''建立模型'''
# coding: utf-8
# import pandas as pd 
import numpy as np 
# import codecs
# import jieba
# import multiprocessing
import keras
import os
import dataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.core import Activation, Dense, Dropout
# from keras.preprocessing import sequence
from keras.models import model_from_yaml
np.random.seed(1337)            #??
import sys
sys.setrecursionlimit(1000000)
import yaml

# CPU_COUNT = multiprocessing.cpu_count() # 4
# VOCAB_DIM = 150         #进行word embbeding操作后得到词向量的长度
# N_EXPOSURES = 5 # 所有频数超过10的词语
# WINDOW_SIZE = 10
# CLASS_NUM = 3       #类别数量   0:差评, 1:中评, 2:好评
# N_ITERATIONS = 1        # ideally more..
# INPUT_LENTH = 150       #句子数字化后对应一位数组的长度, 也是lstm模型输入数据的长度
N_EPOCH = 6
BATCH_SIZE = 64

MODEL_PATH_LSTM = "./model/lstm.h5"

def define_model(embedding_weights):
        """ 定义网络结构
        # param: 
                @embedding_weights: 词嵌入
        # return:
                @model: Sequential模型
        """
        print("define_model... ")
        #定义神经网络模型
        model = Sequential()
        model.add(Embedding(input_dim=embedding_weights.shape[0],
                        output_dim=dataset.VOCAB_DIM,
                        # mask_zero=True,
                        weights=[embedding_weights],
                        input_length=dataset.INPUT_LENTH))
        model.add(keras.layers.core.SpatialDropout1D(0.3))
        model.add(Conv1D(activation='relu', padding='same', filters=64, kernel_size=5))
        model.add(MaxPool1D(pool_size=4))
        model.add(LSTM(128, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='sigmoid'))
        model.add(Activation('softmax'))
        return model


def train_lstm(model, x_train, y_train):
        """ 训练神经网络
        # param: 
                @embedding_weights: 词嵌入
                @x_train, y_train: 训练集输入和标签
        # return:
                None
        """
        print("train_lstm... ")
        #compile模型
        print('Compiling the Model...')
        model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])
        if os.path.isfile(MODEL_PATH_LSTM):
                print('Loading existed model...')
                model.load_weights(MODEL_PATH_LSTM)
        else:
                #训练模型, batch_size = 32
                print('Train...')
                model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCH, verbose=1)
        model.save_weights(MODEL_PATH_LSTM)
        # yaml_string = model.to_yaml()               #yaml仅保存模型的结构, 不包括参数
        # with open('./model/lstm.yml', 'w') as outfile:
                # outfile.write(yaml.dump(yaml_string, default_flow_style = True))
        return model

def train_test_split(x, y, test_size=None, shuffle=True):
        print("train_test_split... ")
        assert x.shape[0] == y.shape[0] and test_size >= 0
        #shuffle, 洗牌, 打乱顺序
        if shuffle:
                perm = np.arange(x.shape[0])
                np.random.shuffle(perm)
                x = x[perm]
                y = y[perm]
        n = int(test_size * x.shape[0])
        if n == 0:
                return x, y, np.array([]), np.array([])
        if test_size == 1:
                return np.array([]), np.array([]), x, y, 
        return x[0: -n], y[0: -n], x[-n: x.shape[0]], y[-n: y.shape[0]]

#训练模型并保存
if __name__ == "__main__":
        print("Loading data... ")
        data_set = dataset.read_data_sets()
        train_sentences = data_set.train.sentences
        train_labels = data_set.train.labels
        embedding_weights = data_set.embedding_weights
        print("sentences_train.shape: %s labels_train.shape: %s"%(train_sentences.shape, train_labels.shape))
        x_train, y_train, x_valid, y_valid  = train_test_split(train_sentences, train_labels, test_size = 0)
        model = define_model(embedding_weights)
        #训练模型
        model = train_lstm(model, x_train, y_train)
        # #验证模型
        # print('Evaluate...')
        # score = model.evaluate(x_valid, y_valid, batch_size=BATCH_SIZE)
        # print("Test score: ", score)






















