#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Utilities for tokenizing text, create vocabulary and so on"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import jieba
import gzip
import os
import re
import tarfile
import pandas as pd
import time

from sklearn.externals import joblib
from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "<pad>"
_GO = "<go>"
_EOS = "<eos>"
_UNK = "<unknown>"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#将原始文章先清洗、分词保存以留备用
# Regular expressions used to tokenize.
#表明原文在处理时似乎是没有去掉标点的;其实原文只做了以下预处理；也没有去停用词
'''
特殊字符：去除特殊字符，如：“「，」,￥,…”；
括号内的内容：如表情符，【嘻嘻】，【哈哈】
日期：替换日期标签为TAG_DATE，如：***年*月*日，****年*月，等等
超链接URL：替换为标签TAG_URL；
删除全角的英文：替换为标签TAG_NAME_EN；
替换数字：TAG_NUMBER；
在对文本进行了预处理后，准备训练语料： 我们的Source序列，是新闻的正文，待预测的Target序列是新闻的标题。
我们截取正文的分词个数到MAX_LENGTH_ENC=120个词，是为了训练的效果正文部分不宜过长。标题部分截取到MIN_LENGTH_ENC = 30，即生成标题不超过30个词
原文：https://blog.csdn.net/rockingdingo/article/details/55224282 
'''
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(rb"\d")

def basic_tokenizer(sentence):
  return sentence.strip().split()
  # #南 都 讯 记!者 刘?凡 周 昌和 任 笑 一 继 推出 日 票 后 TAG_NAME_EN 深圳 今后 将 设 地铁 TAG_NAME_EN 头 等 车厢 TAG_NAME_EN 设 坐 票制
  # """Very basic tokenizer: split the sentence into a list of tokens."""
  # words = []
  # #将每一句的句首和句尾的空白字符(换行符)去掉，然后按空格分割
  # for space_separated_fragment in sentence.strip().split():
  #   print('space_separated_fragment',space_separated_fragment)
  #   #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)
  #   words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  #   print('words',words)#words ['南', '都', '讯', '记', '!', '者', '刘', '?', '凡', '周', '昌和'],取其中某一步的words
  # return [w for w in words if w]#w不为空就返回words中的w,组合成列表sentence_split ['南', '都', '讯', '记', '!', '者', '刘', '?', '凡', '周', '昌和', '任', '笑', '一', '继', '推出',

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  # if not os.path.exists(vocabulary_path):
  if True:
      print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
      vocab = {}  #(词，词频)对
    # with gfile.GFile(data_path, mode="rb") as f:


      # counter = 0

      df=pd.read_parquet(data_path)
      for index,row in df.iterrows():
        line=row['content_split']+' '+row['title_split']
      # for line in f:
      #   counter += 1
        if index % 100000 == 0:
          print("  processing line %d" % index)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for word in tokens:
          # #分词后将每个词中的数字替换为0，如果开启normalize_digits
          # word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w

          #统计每个词以及出现的次数
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      #开始列表相加；字段按照值排序(逆序)后，返回键的列表dict.get(key,default=None)获取键对应的值,default -- 如果指定键的值不存在时，返回该默认值值。
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      #前面一步表示按词频从高到低排列，下一步表示如果词汇量大于50000，则取前50000个词汇
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      all_vocabs_id_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
      print("all_vocabs_id_dict",all_vocabs_id_dict)
      joblib.dump(all_vocabs_id_dict,'all_vocabs_id_dict.pkl')
      # return vocab_list,all_vocabs_id_dict


      # with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      #   for w in vocab_list:
      #     vocab_file.write(w + b"\n")






def word2id_func(path):
    word2id_dic = {}
    all_vocabs_id_dict=joblib.load('all_vocabs_id_dict.pkl')
    data_df=pd.read_parquet(path)
    for index,each_row in data_df.iterrows():
        time_start = time.time()
        print('开始处理第{}行数据'.format(index))
        line=each_row['content_split']+' '+each_row['title_split']
        for each_word in line:
          ######unk
          word2id_dic[each_word]=all_vocabs_id_dict.get(each_word, UNK_ID)

        if index%10000==0:
          time_end = time.time()
          print('10000条数据耗时：{} s'.format(time_end-time_start))
          print('总数据约耗时{} hour'.format(data_df.shape[0]/10000*(time_end-time_start)/3600 ))
          # break
    print(word2id_dic)
    joblib.dump(word2id_dic,'word2id.pkl')
def id2word_func():
    word2id_dic=joblib.load( 'word2id.pkl')
    id2word={v: k for k, v in word2id_dic.items()}
    joblib.dump(id2word, 'id2word.pkl')
    print(id2word)

def trainingSamples_func(path):
    word2id_dic = joblib.load('word2id_dic.pkl')
    print('word2id_dic',word2id_dic)
    print(word2id_dic['本文'])
    trainingSamples=[]
    data_df = pd.read_parquet(path)
    print(data_df.head())


    for index, each_row in data_df.iterrows():
        content_split = each_row['content_split']
        title_split=each_row['title_split']
        #####unk
        content_split_id=[word2id_dic.get(word,UNK_ID) for word in content_split]
        title_split_id=[word2id_dic.get(word,UNK_ID) for word in title_split]
        line=[content_split_id,title_split_id]
        trainingSamples.append(line)
    joblib.dump(trainingSamples, 'trainingSamples.pkl')
    print(trainingSamples)


def final_data():
    all_data={}
    word2id=joblib.load('word2id.pkl')
    id2word=joblib.load('id2word.pkl')
    trainingSamples=joblib.load('trainingSamples.pkl')
    all_data['word2id']=word2id
    all_data['id2word']=id2word
    all_data['trainingSamples']=trainingSamples
    joblib.dump(all_data,'../data/souhu-part3-vocabSize50000.pkl')


if __name__=='__main__':
    data_path='part3_split.parquet.gzip'
    vocabulary_path='.'

    print('create_vocabulary')
    create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=50000,
                    tokenizer=None, normalize_digits=False)
    print('go word2id')
    word2id_func(data_path)
    print('go id2word')
    id2word_func()
    print('go rainingSamples')
    trainingSamples_func(data_path)
    print('go final')
    final_data()
