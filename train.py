import tensorflow as tf

from data_helpers import loadDataset,getBatches, sentence2enco
from data_helpers_new import loadDataset,getBatches, sentence2enco
# from model import Seq2SeqModel
from model_new import Seq2SeqModel
# from model_bidirection import Seq2SeqModel
# from model_bidirection_diff_layer_encoder_decoder import Seq2SeqModel
# from model_bidirection_gru import Seq2SeqModel
# from model_bidirection_copynet import Seq2SeqModel
from tqdm import tqdm
import math
import os
import numpy as np

# http://blog.csdn.net/leiting_imecas/article/details/72367937
# tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 4, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 7, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 5, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

# data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
data_path='data/souhu-part3-vocabSize50000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)
print('word2id',len(word2id))
print('id2word',len(id2word))
'''
word2id 15187
id2word 5000
'''
#id2word以下并没有使用

print('FLAGS.rnn_size',FLAGS.rnn_size)#FLAGS.rnn_size 1024
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
                         max_gradient_norm=5.0
                         )

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')#lhs shape= [50000] rhs shape= [15187]
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    for each in tf.all_variables():
        print('each var',each)
    #显示所有节点名称
    # for n in tf.get_default_graph().as_graph_def().node:
    #     print('n.name',n.name)



    # current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples, FLAGS.batch_size)
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for nextBatch in tqdm(batches, desc="Training"):
            #padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3
            print('nextBatch',np.array(nextBatch.decoder_targets).shape,'\n',)#np.array(nextBatch.decoder_targets))
            print('nextBatch.decoder_targets_length',nextBatch.decoder_targets_length)


            loss, summary,current_step = model.train(sess, nextBatch)
            print('loss',loss)

            logist_, label_,decoder_inputs_ = model.train(sess, nextBatch)


            # print('logist_,label_,decoder_inputs_',np.array(logist_).shape,'\n',np.array(label_).shape,np.array(decoder_inputs_).shape,np.array(logist_),np.array(label_),np.array(decoder_inputs_))

            # current_step += 1
            # 每多少步进行一次保存
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)


'''
data_helpers
nextBatch (7, 51) 
 [[ 126    3 2463 3340    3    3    3    3    3   45    3 2596    3    3
   342  186    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]
 [ 846    3  877    3   20    3    3   68 1315 1687    3   13    3  141
    18    3 1440  362    3    4    3   40   20 3585    3    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]
 [   3    3    3    3    3    3    3    3    3    3  171    3    3    3
   699  589    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]
 [   3 1054 1937    3 1565    3 2130  614    3    3    3    3  152 1821
     3 1154  130    3    3    3    3    3 1054 1937    3 1166 1054    3
     3    3 3499 4232    3 3771 1359    3 2011  301    3    3 1054 1937
     3 1565    3 2130  614    3    3    3   96]
 [ 664    3    3    3    3    3  362    3    3    4    3    3    3    3
     3    3 3337 1519    3  877    3    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]
 [   3    3    3    3    3    3    3    3    3    7    3   70   20   18
  2015    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]
 [ 169 1519  583    3   16   68    3 1710   22    3    3  583    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0]]

#data_helpers_new    
     
InvalidArgumentError (see above for traceback): logits and labels must have the same first dimension, 
got logits shape [357,15187] and labels shape [364]
357=batch_size*time_step=7*51
364=7*52

	 [[node decoder/sequence_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits (defined at /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/contrib/seq2seq/python/ops/loss.py:92)  = SparseSoftmaxCrossEntropyWithLogits[T=DT_FLOAT, Tlabels=DT_INT32, _device="/job:localhost/replica:0/task:0/device:CPU:0"](decoder/sequence_loss/Reshape, decoder/sequence_loss/Reshape_1)]]

nextBatch (7, 52) 
 [[ 126    3 2463 3340    3    3    3    3    3   45    3 2596    3    3
   342  186    2    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]
 [ 846    3  877    3   20    3    3   68 1315 1687    3   13    3  141
    18    3 1440  362    3    4    3   40   20 3585    3    2    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]
 [   3    3    3    3    3    3    3    3    3    3  171    3    3    3
   699  589    2    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]
 [   3 1054 1937    3 1565    3 2130  614    3    3    3    3  152 1821
     3 1154  130    3    3    3    3    3 1054 1937    3 1166 1054    3
     3    3 3499 4232    3 3771 1359    3 2011  301    3    3 1054 1937
     3 1565    3 2130  614    3    3    3   96    2]
 [ 664    3    3    3    3    3  362    3    3    4    3    3    3    3
     3    3 3337 1519    3  877    3    2    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]
 [   3    3    3    3    3    3    3    3    3    7    3   70   20   18
  2015    2    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]
 [ 169 1519  583    3   16   68    3 1710   22    3    3  583    2    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0]]

'''