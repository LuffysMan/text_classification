提交结果文件格式：
参赛者需要提交预测结果文件，文件每一行对应一个预测结果：id，label
提交样例：
1,0
2,1
3,3
......
注意：
（1）提交结果共两列，第一列为数据id，第二列为预测结果（label），字段间以逗号为分割符，提交结果问价不包含列名（id，label）；
（2）提交结果的行数要求与测试数据一致，且顺序一致；
（3）提交结果文件的格式要求为csv；
（4）不符合格式要求的提交结果，将被评为无效成绩，以“-”标识。

您的好友邀请您组队参赛，详情查看：https://dianshi.baidu.com/invite?code=vv5xP4

参考博客:
LSTM三分类: https://github.com/Edward1Chou/SentimentAnalysis/blob/master/lstm/lstm_train.py#L125
基于循环神经网络(LSTM)的文本情感分类: https://blog.csdn.net/smilejiasmile/article/details/80952438
gensim学习之corpora.Dictionary: http://blog.sina.com.cn/s/blog_6877dad30102xc7n.html
PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
Keras入门（二）模型的保存、读取及加载: https://blog.csdn.net/jclian91/article/details/83038861
Keras中predict()方法和predict_classes()方法的区别: https://blog.csdn.net/tszupup/article/details/85275111
keras系列︱keras是如何指定显卡且限制显存用量（GPU/CPU使用）: https://blog.csdn.net/sinat_26917383/article/details/75633754
keras深度训练4:GPU设置: https://blog.csdn.net/github_36326955/article/details/79910448
Numpy之结构数组与数组持久化:https://www.cnblogs.com/AllStarGIS/p/3782748.html
[神经网络]keras中的层: https://blog.csdn.net/pengjian444/article/details/56316445

关于交叉熵在loss函数中使用的理解:https://blog.csdn.net/tsyccnh/article/details/79163834
关于sigmoid与binary_crossentropy，以及softmax与categorical_crossentropy的关系，以及各损失函数的定义。:https://blog.csdn.net/u010995990/article/details/79739887


遇到的问题总结:
问题1: pandas read_csv()报错:Python pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 121, saw 2
解决: 设置正确参数(encoding, sep), 参考https://www.jianshu.com/p/be233bdb4dbf

问题2: 
python -- lambda表达式, 参考https://www.cnblogs.com/hf8051/p/8085424.html,
Python中的lambda和apply用法,参考https://blog.csdn.net/anshuai_aw1/article/details/82347016
Python3下map函数的显示问题, 参考https://www.cnblogs.com/itdyb/p/5731804.html

问题:代码中使用了model.vocab.keys(), 报错AttributeError: 'Word2Vec' object has no attribute 'vocab'
原因:gensim 1.0以后的版本, 将model.vocab移动到了model.wv.vocab
解决: model.wv.vocab.keys()
参考: https://stackoverflow.com/questions/42517435/gensim-word2vec-in-python3-missing-vocab

问题: 使用model.save_weights()保存神经网络参数后, 在后续进行预测时, 直接定义了一个空的model,
model = Sequential()
model.load_weights("./model/weights.model")
model.predict(x_test)
报错
原因: 只是保存了模型的参数, 加载之前需定义好模型结构, 定义成保存前一摸一样的结果即可, 并调用model.compile()
然后就可以调用
model.predict(x_test)

问题: keras指定显卡
解决: CUDA_VISIBLE_DEVICES=1 python train.py
CUDA_VISIBLE_DEVICES: 显卡编号, 可以通过nvidia-smi 进行查看



第一阶段: 初级阶段, 实现lstm神经网络, 训练模型, 验证模型
完成
第二阶段: 优化阶段, 提交数据集到比赛官网, 查看分数, 然后优化算法
可以从以下几个方面来考虑优化:
1. 代码结构优化: 包括把特征提取, 模型训练, 模型测试模块话, 代码尽可能复用, 代码逻辑清晰等等, 减少代码方面的bug和消耗
1. 特征提取: 包括停用词表, 句子的最大长度, 词向量模型(word to vector? TF-IDF? 还是其他), padding在后面补零?
2. 神经网络优化: 包括正则化, 激活函数选择, 调整神经网络节点数和层数, 使用滑动平均模型, 使用drop-out, 参数初始化,
使用偏移

原模型:
输入句子长度100, 经过word embedding层, 变为120, 
lstm层50, 采用tahn激活
验证集准确率 0.92
提交测试分数 0.9057

尝试1:
输入句子长度200, word embedding层: vocab_size x 150
lstm层300 采用relu激活
出错: 在输入大约7000左右个batch的时候, loss变为nan, 初步判断是有脏数据导致交叉熵计算log时输入为0或负数
在尝试2中改了几个参数, 感觉应该是吧relu改为tanh立功了

尝试2:
输入句子长度200, word embedding层: vocab_size x 150
lstm层128 采用tanh激活
validate score:  [0.64039096114331, 0.907832977754319]
test socre: 0.9017

尝试3:
数据预处理时, 把评论类型加入到输入当中
在lstm层之前增加卷积池化层
参考 神经网络优化_dropout_正则化_cnn特征提取: https://blog.csdn.net/Xwei1226/article/details/81582777
test socre: 0.8968


尝试4:
增加训练集, 减少验证集
修改激活函数: dense(3, activation='sigmoid)
追加训练轮数, 4轮后再加2轮
test socre: 0.9115


尝试5:
去掉验证集, 全部数据用来训练, 训练6轮
进行结巴分词的时候, 仅去掉标点符号
test socre: 	0.9291

预尝试6:
数据集:
    分词: 只去掉标点
    句子长度: 200
    vocab_size: 150
    坏数据填充: '空'
模型结构:
       model.add(Embedding(...))
        model.add(keras.layers.core.SpatialDropout1D(0.3))
        #加入卷积池化
        model.add(Conv1D(activation='tanh', padding='same', filters=64, kernel_size=5))
        model.add(MaxPool1D(pool_size=4))
        #双向lstm
        model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), merge_mode='concat'))
        model.add(Dropout(0.3))
        model.add(keras.layers.Flatten())
        model.add(Dense(3, activation='tanh'))
        model.add(Activation('softmax'))
train socre: 	0.9554
test socre: 	0.9302

尝试7:
数据集:
    分词: 只去掉标点
    句子长度: 200
    vocab_size: 150
    坏数据填充: '空'
模型结构:
         #定义神经网络模型
        model = Sequential()
        model.add(Embedding(input_dim=embedding_weights.shape[0],
                        output_dim=dataset.VOCAB_DIM,
                        # mask_zero=True,
                        weights=[embedding_weights],
                        input_length=dataset.INPUT_LENTH))
        model.add(keras.layers.core.SpatialDropout1D(0.3))
        model.add(Conv1D(activation='tanh', padding='same', filters=128, kernel_size=5))
        model.add(MaxPool1D(pool_size=4))
        model.add(Bidirectional(LSTM(32, return_sequences=True, activation='tanh'), merge_mode='concat'))
        model.add(Dropout(0.3))
        model.add(keras.layers.Flatten())
        model.add(Dense(3, activation='tanh', 
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01)))
        model.add(Activation('softmax'))
train socre: 	0.9616
test socre: 	0.9314

尝试8:
数据集:
    分词: 只去掉标点
    句子长度: 200
    vocab_size: 150
    坏数据填充: '空'
模型:
    model = Sequential()
    model.add(Embedding(input_dim=embedding_weights.shape[0],
                    output_dim=dataset.VOCAB_DIM,
                    # mask_zero=True,
                    weights=[embedding_weights],
                    input_length=dataset.INPUT_LENTH))
    model.add(keras.layers.core.SpatialDropout1D(0.3))
    model.add(Conv1D(activation='tanh', padding='same', filters=32, kernel_size=5))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(50, activation='tanh'))
    # model.add(keras.layers.Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(3, activation='tanh'))
    model.add(Activation('softmax'))
test socre: 	0.9331




如何判断优化的方向:
出现过拟合的时候，虽然loss训练误差在每个Epoch稳步下降，但是在测试集上val_loss没有下降，反而有上升的趋势，
如果val_loss比loss高出很多，这说明模型已经严重过拟合了


总结:
使用卷积层大大加快了训练, 而且提升了准确率;
不同的激活函数效果差别大



