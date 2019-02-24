#coding: utf-8
#预处理数据, 封装成方便使用的数据集
import pandas as pd
import numpy as np 
import keras
import jieba
import codecs
import os
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
#from nltk.corpus import stopwords                       #加载停用词表, 没有中文停用词表
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

CPU_COUNT = multiprocessing.cpu_count() # 4
VOCAB_DIM = 150         #进行word embbeding操作后得到词向量的长度
N_EXPOSURES = 5 # 所有频数超过10的词语
WINDOW_SIZE = 10
N_ITERATIONS = 1        # ideally more..
NUM_CLASSES = 3       #类别数量   0:差评, 1:中评, 2:好评
INPUT_LENTH = 200

MODEL_PATH_WV = "./model/Word2vec_model.pkl"
FILE_TRAIN_CSV = 'dataset/data_train.csv'
FILE_TEST_CSV = 'dataset/data_test.csv'
FILE_TRAIN_TMP = 'dataset/data_train.npy'
FILE_TEST_TMP = 'dataset/data_test.npy'
FILE_EMB_TMP = 'dataset/emb_weights.npy'
#载入停用词表
stopwords = codecs.open("./stopwords.txt", encoding='utf-8').read().split('\n')
punctuation = codecs.open("./punctuation.txt", encoding='utf-8').read().split('\n')

def loadfile(filepath):
    #读取csv训练数据, 返回评论内容和类别标签
    print("loadfile...")
    df = pd.read_csv(filepath, header=None, encoding='gbk', sep='\t')
    concatenate = lambda x, y: str(x)+str(y)
    type_com = [concatenate(df[1][i], df[2][i]) for i in range(len(df))]
    comments = np.array(type_com)
    try:
        labels = np.array(df[3])    
    except:
        labels = np.zeros((len(comments)))
    '''    y = np.zeros((num_samples, CLASS_NUM))  #得到标签one-hot向量
    for i in range(num_samples):    
    j = df[3][i]
    y[i][j] = 1'''
    return comments,labels

def segment(comments):
    #对评论句子进行分词
    #seg = lambda x: [word for word in jieba.cut(x.replace('\n', '')) if word not in stopwords and word.strip() != '']
    print("segment...")
    def seg(x): #闭包
        try:
                #words =[word for word in jieba.cut(x.replace('\n', '')) if word not in stopwords and word.strip() != '']
                words =[word for word in jieba.cut(x.replace('\n', '')) if word not in punctuation and word.strip() != '']
        except:
                words = ['空']   #遇到坏数据以"空"代替
                print("bad data: ", x)
        return words
    comments_segmented = [seg(comment) for comment in comments]
    return comments_segmented

def create_dictionaries(model=None, comments_seged=None):
    """ Function does are number of Jobs:
    1- Creates a word to index mapping
    2- Creates a word to vector mapping
    3- Transforms the Training and Testing Dictionaries
    param:
    @model: Word2vec()处理后的模型, 
    @comments_segmented: 经过分词处理后的评论
    return: 
    @dct_w2indx: 字典{词语, id}
    @dct_w2vec: 字典{词语, 词向量}
    @comments_segmented_numberized: 列表[[词id, 词id2], [], ...]
    """
    print("create_dictionaries...")
    if (comments_seged is not None) and (model is not None):
            gensim_dict = Dictionary()    #gensim_dict:  Dictionary(647 unique tokens: ['app', '一套', '久', '买', '代']...)
            gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True) #打印看看, vocab.keys(): [单词1, 单词2, 单词3]
            #gensim_dict:  [(19, 1), (29, 1), (32, 1), (45, 1), (102, 1), (108, 1), (130, 1)]
            #频数小于10的词的索引为0, 所以k+1
            #dct_w2indx = gensim_dict.token2id                        #字典{词语:词频}
            dct_w2indx = {v: k+1 for k, v in gensim_dict.items()}     #所有频数超过10的词语的索引,(k->v)=>(v->k), {'hellip': 1, '上菜': 2, '不错': 3, ...}
            #dct_w2vec = {word: model[word] for word in dct_w2indx.keys()}   #所有频数超过10的词语的词向量构成一个向量字典, (word->model(word))
            #array = np.array(list(dct_w2vec.values()))
            #print("shape: ", np.shape(array))              #shape:  (32, 120)
            def parse_dataset(setences_segmented):#闭包, 临时使用, 将由分词构成的句子表示成有词语id构成的句子, 每个句子是一个list
                    setences_numberized = []
                    for setence in setences_segmented:
                            setence_numberized = []
                            for word in setence:
                                    try:
                                            setence_numberized.append(dct_w2indx[word])
                                    except:
                                            setence_numberized.append(0)
                            setences_numberized.append(setence_numberized)
                    return setences_numberized
            comments_seged_numbered = parse_dataset(comments_seged)       #将分词后的评论集合数字化
            comments_seged_numbered = sequence.pad_sequences(comments_seged_numbered, maxlen=INPUT_LENTH) #句子统一为固定长度
            return np.array(comments_seged_numbered)
    else:
            print("No data provided")

def create_emb_weights(model=None):
    print("create_emb_weights...")
    if model is not None:
        gensim_dict = Dictionary()    #gensim_dict:  Dictionary(647 unique tokens: ['app', '一套', '久', '买', '代']...)
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True) #打印看看, vocab.keys(): [单词1, 单词2, 单词3]
        #gensim_dict:  [(19, 1), (29, 1), (32, 1), (45, 1), (102, 1), (108, 1), (130, 1)]
        #频数小于10的词的索引为0, 所以k+1
        #dct_w2indx = gensim_dict.token2id                        #字典{词语:词频}
        dct_w2indx = {v: k+1 for k, v in gensim_dict.items()}     #所有频数超过10的词语的索引,(k->v)=>(v->k), {'hellip': 1, '上菜': 2, '不错': 3, ...}
        dct_w2vec = {word: model[word] for word in dct_w2indx.keys()}   #所有频数超过10的词语的词向量构成一个向量字典, (word->model(word))
        n_symbols = len(dct_w2indx) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, VOCAB_DIM))    #初始化 索引为0的词语，词向量全为0 
        for word, index in dct_w2indx.items():  # 从索引为1的词语开始，对每个词语对应其词向量??
                embedding_weights[index, :] = dct_w2vec[word]
        print("embedding_weights: ", np.shape(embedding_weights))
        return embedding_weights 
    else:
        return None 

def word2vec_train(comments_segmented = None):
    ''' 训练一个word2vec模型
    param:
    @comments: 评论列表, list, [[评论1], [评论2], ...]
    return:
    @dct_w2indx: 字典{词语: id}
    @dct_w2vec: 字典{词语: 词向量}
    @comments_segmented_numberized: 向量化的评论, 列表[[词向量1, 词向量2], [], ...]
    '''
    print("word2vec_train...")
    model = Word2Vec(size=VOCAB_DIM, 
                    min_count=N_EXPOSURES, 
                    window=WINDOW_SIZE,
                    workers=CPU_COUNT,
                    iter=N_ITERATIONS)
    if os.path.isfile(MODEL_PATH_WV):       #词向量模型已存在, 直接加载
            model = Word2Vec.load(MODEL_PATH_WV)
    else:
            model.build_vocab(comments_segmented)     #输入格式: list
            model.train(comments_segmented, total_examples=len(comments_segmented), epochs=model.epochs)
            model.save(MODEL_PATH_WV)
    return model

def create_bin():
    print("create_bin...")
    #生成并保存训练集和词嵌入向量的二进制文件
    comments, labels = loadfile(FILE_TRAIN_CSV)
    labels = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    comments_seged = segment(comments)
    modelwv = word2vec_train(comments_seged)    #训练word2vec模型, 并保存
    comments_seged_numbered = create_dictionaries(model=modelwv, comments_seged=comments_seged)
    assert comments_seged_numbered.shape[0] == labels.shape[0]
    train_sets = np.concatenate((comments_seged_numbered, labels), axis=1)
    train_sets = np.array(train_sets, np.float)
    print("train_sets: ", train_sets.shape, train_sets[0])
    # train_sets.tofile(FILE_TRAIN_BIN)
    np.save(FILE_TRAIN_TMP, train_sets)
    #生成词嵌入向量并保存
    embedding_weights = create_emb_weights(model=modelwv)
    embedding_weights = np.array(embedding_weights, np.float)
    # embedding_weights.tofile(FILE_EMB_BIN)
    np.save(FILE_EMB_TMP, embedding_weights)
    #生成并保存测试集的二进制文件
    comments, labels = loadfile(FILE_TEST_CSV)
    labels = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    comments_seged = segment(comments)
    modelwv = word2vec_train(comments_seged)    #加载word2vec模型
    comments_seged_numbered = create_dictionaries(model=modelwv, comments_seged=comments_seged)
    assert comments_seged_numbered.shape[0] == labels.shape[0]
    test_sets = np.concatenate((comments_seged_numbered, labels), axis=1)
    test_sets = np.array(test_sets, np.float)
    # test_sets.tofile(FILE_TEST_BIN)
    np.save(FILE_TEST_TMP, test_sets)
    # np.savez('dataset/data_sets.npz', train_sets, test_sets, embedding_weights)
    return train_sets, test_sets, embedding_weights

def get_trainset():
    print("get_trainset...")
    data_sets = np.array([])
    if not os.path.isfile(FILE_TRAIN_TMP):
        data_sets, _, _ = create_bin()
    #若句子数字化已经保存到文件中, 直接加载
    data_sets = np.load(FILE_TRAIN_TMP)
    sentences = data_sets[:, 0: -3]
    labels = data_sets[:, -3: data_sets.shape[1]]
    return sentences, labels

def get_testset():
    print("get_testset...")
    data_sets = np.array([])
    if not os.path.isfile(FILE_TEST_TMP):
        _, data_sets, _ = create_bin()
    #若句子数字化已经保存到文件中, 直接加载
    data_sets = np.load(FILE_TEST_TMP)
    sentences = data_sets[:, 0: -3]
    labels = data_sets[:, -3: data_sets.shape[1]]
    return sentences, labels

def get_emb_weights():
    print("get_emb_weights...")
    emb_weights = np.array([])
    if not os.path.isfile(FILE_EMB_TMP):
        _, _, emb_weights = create_bin()
    #若句子数字化已经保存到文件中, 直接加载
    emb_weights = np.load(FILE_EMB_TMP)
    return emb_weights

class DataSet(object):
    def __init__(self, sentences, labels, fake_data=False):
        if fake_data:
            self.num_examples = 10000
        else:
            assert sentences.shape[0] == labels.shape[0] ,(
                "sentences.shape: %s labels.shape:%s"%(sentences.shape, labels.shape))
            self._num_examples = sentences.shape[0]
        self._sentences = sentences
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def sentences(self):
        return self._sentences
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        """Return the next 'batch_size' data from this data set."""
        if fake_data:
            fake_sentence = [1.0 for _ in range(200)]
            fake_label = 0
            return [fake_sentence for _ in range(batch_size)], [fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            #Finished epoch 
            self.epochs_completed += 1
            #Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._sentences = self._sentences[perm]
            self._labels = self._labels[perm]
            #Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size < self._num_examples
        end = self._index_in_epoch
        return self._sentences[start: end], self._labels[start: end]

def read_data_sets(fake_data=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.sentences = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.labels = DataSet([], [], fake_data=True)
        return data_sets
    train_sentences, train_labels = get_trainset()
    test_sentences, test_labels = get_testset() 
    data_sets.embedding_weights = get_emb_weights()
    data_sets.train = DataSet(train_sentences, train_labels)
    data_sets.test = DataSet(test_sentences, test_labels)
    return data_sets












