#coding: utf-8
#使用模型对测试集进行预测, 按照id+label输出测试结果到csv文件, 用于提交
import keras
import keras.models
import numpy as np 
import pandas as pd
import lstm_model
import dataset

# from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.preprocessing import sequence
from keras.models import model_from_yaml

OUTPUT_FILE = "./output/data_test_pre.csv"

def save_csv(y_pre):
    #保存预测后的结果到csv文件, 保护两列[id, label]
    print("Saving prediction... ")
    id_label = np.array([[k+1, y_pre[k]] for k in range(len(y_pre))])
    print("id_label: ", id_label.shape)
    df = pd.DataFrame(id_label)
    df.to_csv(OUTPUT_FILE, header=None, index=False)

if __name__ == "__main__":
    print("Loading test data...")   #加载测试集
    data_set = dataset.read_data_sets()
    x_test = data_set.test.sentences
    embedding_weights = data_set.embedding_weights
    print("x_test.shape: ",x_test.shape)
    print("Loading lstm model...")
    model = lstm_model.define_model(embedding_weights)
    model.load_weights(lstm_model.MODEL_PATH_LSTM)
    print("Predicting...")
    y_pre = model.predict_classes(x_test)
    save_csv(y_pre)































