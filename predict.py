import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np


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

data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)
#预测时用到id2word
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

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:

        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=True, beam_size=20, max_gradient_norm=5.0)
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
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        print('batch',batch)
        # 获得预测的id
        predicted_ids = model.infer(sess, batch)
        print('predicted_ids',np.array(predicted_ids).shape)
        # 将预测的id转换成汉字
        predict_ids_to_seq(predicted_ids, id2word, 5)
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
