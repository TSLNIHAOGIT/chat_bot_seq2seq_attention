import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np
from tqdm import tqdm
import math


tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

# data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
data_path='data/souhu-part2-vocabSize50000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)
print('loadDataset')
# print(word2id)
# '''{'<pad>': 0, '<go>': 1, '<eos>': 2, '<unknown>': 3, 'can': 4, 'we': 5, 'make': 6,'''
# print(id2word)
# '''{0: '<pad>', 1: '<go>', 2: '<eos>', 3: '<unknown>', 4: 'can', 5: 'we', 6: 'make', 7: 'this', 8: 'quick'''

print(np.array(trainingSamples).shape)#(159657, 2)
'''
trainingSamples[0:3]
 是一个二维数组，形状为N*2，每一行包含问题和回答
[ 
 [[40, 22], [50, 9, 51, 9]], 
 [[57, 33, 58, 59, 23, 9],[60, 61, 22]],
 [[73, 22],[63, 84, 22]]
]
'''



with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size,
                         FLAGS.num_layers,
                         FLAGS.embedding_size,
                         FLAGS.learning_rate,
                         word2id,
                         mode='train',
                         use_attention=True,
                         beam_search=False,
                         beam_size=5,
                         max_gradient_norm=5.0)
    # path_temp='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/tensorflow_practice_from_git/nlp/chat_bot_seq2seq_attention/model'
    ckpt = tf.train.get_checkpoint_state(
        # path_temp
        FLAGS.model_dir
    )
    print('FLAGS.model_dir',FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

    for each in tf.all_variables():
        print('each',each)

    for e in range(FLAGS.numEpochs):
        # print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples, FLAGS.batch_size)
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for nextBatch in tqdm(batches, desc="Training"):
            loss_, summary_ = model.eval(sess, nextBatch)
            perplexity = math.exp(float(loss_)) if loss_ < 300 else float('inf')
            print('验证集上loss为:{},perplexity为:{}'.format(loss_,perplexity))
        break
