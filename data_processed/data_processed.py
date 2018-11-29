import pandas as pd
import jieba
from sklearn.externals import joblib
import time
import tqdm
jieba.enable_parallel()
# print(word2id)
# '''{'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3, 'can': 4, 'we': 5, 'make': 6,'''
# print(id2word)
# '''{0: '<pad>', 1: '<go>', 2: '<eos>', 3: '<unknown>', 4: 'can', 5: 'we', 6: 'make', 7: 'this', 8: 'quick'''
'''
trainingSamples[0:3]
 是一个二维数组，形状为N*2，每一行包含问题和回答
[ 
 [[40, 22], [50, 9, 51, 9]], 
 [[57, 33, 58, 59, 23, 9],[60, 61, 22]],
 [[73, 22],[63, 84, 22]]
]

:param filename: 数据的路径，数据是一个json结构，包含三部分，分别是word2id，即word到id的转换，
id2word，即id到word的转换 ，以及训练数据trainingSamples，是一个二维数组，形状为N*2，每一行包含问题和回答
:return: 通过pickle解析我们的数据，返回上述的三部分内容。
    
'''
#
''' content        title'''
path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/seq2seq-chinese-textsum/news/lcsts_part1.parquet.gzip'
def word2id_func():
    data_df=pd.read_parquet(path)
    print('data_df.shape',data_df.shape)#(2400591, 2)
    print(data_df.head())
    word2id_dic={'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3}
    i=len(word2id_dic)

    for index,each_row in data_df.iterrows():
        time_start = time.time()
        print('开始处理第{}行数据'.format(index))
        content=each_row['content']
        title = each_row['title']

        content_split=jieba.cut(content)#返回生成器
        title_split = jieba.cut(title)


        for each in content_split:
            if each not in word2id_dic:
                word2id_dic[each]=i
                i=i+1
        for each in title_split:
            if each not in word2id_dic:
                word2id_dic[each]=i
                i=i+1
        if index==10000:
          time_end = time.time()
          print('10000条数据耗时：{} s'.format(time_end-time_start))
          print('总数据约耗时{} hour'.format(data_df.shape[0]/10000*(time_end-time_start)/3600 ))
          # break



    print(word2id_dic)
    joblib.dump(word2id_dic,'word2id_dic.pkl')
def id2word_func():
    word2id_dic=joblib.load( 'word2id_dic.pkl')
    id2word={v: k for k, v in word2id_dic.items()}
    joblib.dump(id2word, 'id2word.pkl')
    print(id2word)

def trainingSamples_func():
    word2id_dic = joblib.load('word2id_dic.pkl')
    print('word2id_dic',word2id_dic)
    print(word2id_dic['本文'])
    trainingSamples=[]
    data_df = pd.read_parquet(path)
    print(data_df.head())


    for index, each_row in data_df.iterrows():
        content = each_row['content']
        title=each_row['title']
        content_split = jieba.cut(content)
        title_split=jieba.cut(title)
        content_split_id=[word2id_dic[word] for word in content_split]
        title_split_id=[word2id_dic[word] for word in title_split]
        line=[content_split_id,title_split_id]
        trainingSamples.append(line)
    joblib.dump(trainingSamples, 'trainingSamples.pkl')
    print(trainingSamples)


def final_data():
    all_data={}
    word2id_dic=joblib.load('word2id_dic.pkl')
    id2word=joblib.load('id2word.pkl')
    trainingSamples=joblib.load('trainingSamples.pkl')
    all_data['word2id']=word2id_dic
    all_data['id2word']=id2word
    all_data['trainingSamples']=trainingSamples
    joblib.dump(all_data,'../data/dataset-cornell-length10-filter1-vocabSize40000.pkl')


if __name__=='__main__':
    word2id_func()
    id2word_func()
    trainingSamples_func()
    final_data()
