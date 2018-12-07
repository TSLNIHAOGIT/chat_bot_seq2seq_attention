import tensorflow as tf
from tensorflow.python.util import nest


class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 beam_search, beam_size, max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        #执行模型构建部分的代码
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size*2)#原来(单层)是没有乘以2的，大小要与state中h或者c的第二个维度大小一样
            #添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        #列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    #此处只定义了前向和后向的lstm，组合成双向和堆叠成多层要放在后面进行
    # def _create_bi_rnn_cell(self):
    def bi_single_rnn_cell(self):
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell_fw = tf.contrib.rnn.LSTMCell(self.rnn_size)
            single_cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_size)
            #添加dropout
            cell_drop_fw = tf.contrib.rnn.DropoutWrapper(single_cell_fw, output_keep_prob=self.keep_prob_placeholder)
            cell_drop_bw = tf.contrib.rnn.DropoutWrapper(single_cell_bw, output_keep_prob=self.keep_prob_placeholder)
            return cell_drop_fw,cell_drop_bw



    def build_model(self):
        print('building model... ...')
        #=================================1, 定义模型的placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
        #  tf.sequence_mask([1, 3, 2], 5)
        #  [[True, False, False, False, False],
        #  [True, True, True, False, False],
        #  [True, True, False, False, False]]
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        ##=================================2, 定义模型的encoder部分
        # with tf.variable_scope('encoder'):
        #     #创建LSTMCell，两层+dropout
        #     encoder_cell = self._create_rnn_cell()
        #     #构建embedding矩阵,encoder和decoder公用该词向量矩阵
        #     embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
        #     encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            # encoder_outputs用于attention，batch_size*encoder_inputs_length*rnn_size,
            # encoder_state用于decoder的初始化状态，batch_size*rnn_szie
            # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
            #                                                    sequence_length=self.encoder_inputs_length,
            #                                                    dtype=tf.float32)

            #此处用多层堆叠的双向lstm



        with tf.variable_scope('encoder'):

            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            _inputs = encoder_inputs_embedded
            print('_inputs',_inputs)#Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
            ###
            #5-50／10000 L2；


            if len(_inputs.get_shape().as_list()) != 3:
                raise ValueError("the inputs must be 3-dimentional Tensor")
            all_layer_final_state=[]
            for index,_ in enumerate(range(self.num_layers)):
                # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
                # 恶心的很.如果不在这加的话,会报错的.
                 with tf.variable_scope(None, default_name="bidirectional-rnn"):
                    print(index, '_inputs o', _inputs)
                    '''
                    0 _inputs o Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
                    1 _inputs o Tensor("encoder/concat:0", shape=(?, ?, 2048), dtype=float32)
                    2 _inputs o Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)
                    '''


                    #这个结构每次要重新加载，否则会把之前的参数也保留从而出错
                    rnn_cell_fw, rnn_cell_bw = self.bi_single_rnn_cell()


                    initial_state_fw = rnn_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
                    initial_state_bw = rnn_cell_bw.zero_state(self.batch_size, dtype=tf.float32)
                    (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, sequence_length=self.encoder_inputs_length,
                                                                      initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                                                      dtype=tf.float32)

                    print('index,output',index,output)
                    '''
                    output输出每次都是1024，不管输入是多少维度的，相当与接了1024的全链接层
                    index,output 0 (<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
                    index,output 1 (<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn_1/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
                    index,output 2 (<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn_2/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)

                    '''
                    print('''type state[0].c''',type(state[0].c))
                    #type state[0].c <class 'tensorflow.python.framework.ops.Tensor'>

                    _inputs = tf.concat(output, 2)
                    encoder_final_state_c = tf.concat(
                        (state[0].c, state[1].c), 1)

                    encoder_final_state_h = tf.concat(
                        (state[0].h, state[1].h), 1)

                    encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
                        c=encoder_final_state_c,
                        h=encoder_final_state_h
                    )
                    all_layer_final_state.append(encoder_final_state)





                    print('index:{},state:{}'.format(index,state))
                    '''
                    self.num_layers=3时
                    index:0,state:(
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))
 
                    index:1,state:(
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

                    index:2,state:(
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), 
                    LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

                    
                    '''



            encoder_outputs =_inputs
            encoder_state=tuple(all_layer_final_state)#state#
            print('encoder_state',encoder_state)
            '''
            encoder_state (
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/concat_2:0' shape=(?, 2048) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_2:0' shape=(?, 2048) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_2:0' shape=(?, 2048) dtype=float32>),
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_2:0' shape=(?, 2048) dtype=float32>))

            '''


            print('encoder_outputs',encoder_outputs)
            '''
            encoder_outputs Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)

            '''


            print('state',state)#
            '''
            state (
             LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>),
             LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))

            '''


            #encoder_state，应该为每一层最后时刻的lstmtuple()组成的tuple
            print('encoder_state',encoder_state)
            '''
            (
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/concat_2:0' shape=(?, 2048) dtype=float32>), 
            LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_1:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_2:0' shape=(?, 2048) dtype=float32>))

            '''


        # =================================3, 定义模型的decoder部分
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            print('decoder encoder_outputs',encoder_outputs)
            #decoder encoder_outputs Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)

            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                     memory_sequence_length=encoder_inputs_length)
            print('attention_mechanism',attention_mechanism)
            #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self._create_rnn_cell()#维度要与状态维度一致
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            # batch_size = self.batch_size
            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值;decoder多层时每一层都要进行初始化，只是这里选择了用encoder层所有层最后时刻的状态进行初始化；估计换成其它的也可以
            #decoder_initial_state的hun_hidden维度与h或者c的第二维度一样；因为clone(cell_state=encoder_state,shape=(?, 2048))所以hun_hidden也要为2048
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            print('decoder_initial_state',decoder_initial_state)
            '''
            decoder_initial_state :4层的lstm,每一层都要初始化
            AttentionWrapperState(
              cell_state=(
              LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/concat_1/Identity:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/concat_2/Identity:0' shape=(?, 2048) dtype=float32>), 
              LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_1/Identity:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/concat_2/Identity:0' shape=(?, 2048) dtype=float32>),
              LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_1/Identity:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_2/concat_2/Identity:0' shape=(?, 2048) dtype=float32>), 
              LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_1/Identity:0' shape=(?, 2048) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_3/concat_2/Identity:0' shape=(?, 2048) dtype=float32>)),
               attention=<tf.Tensor 'decoder/AttentionWrapperZeroState/zeros_2/Identity:0' shape=(?, 1024) dtype=float32>, time=<tf.Tensor 'decoder/AttentionWrapperZeroState/zeros_1:0' shape=() dtype=int32>, 
               alignments=<tf.Tensor 'decoder/AttentionWrapperZeroState/zeros/Identity:0' shape=(?, ?) dtype=float32>, alignment_history=(), attention_state=<tf.Tensor 'decoder/AttentionWrapperZeroState/zeros_3/Identity:0' shape=(?, ?) dtype=float32>)

            
            '''



            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
                # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                #训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功能
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False, name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=decoder_initial_state, output_layer=output_layer)
                #调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                    maximum_iterations=self.max_target_sequence_length)
                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, weights=self.mask)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()
                #自己加的
                self.global_step = tf.Variable(0, trainable=False)  # 设置global_step为不可训练数值，在训练过程中它不进行相应的更新
                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params),global_step=self.global_step)
            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<go>']
                end_token = self.word_to_idx['<eos>']
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens, end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                maximum_iterations=10)
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        # =================================4, 保存模型
        self.saver = tf.train.Saver(tf.global_variables(),
        # self.saver = tf.train.Saver(tf.trainable_variables(),
         max_to_keep = 5

                                    )

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.5,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary,step = sess.run([self.train_op, self.loss, self.summary_op,self.global_step], feed_dict=feed_dict)
        return loss, summary,step

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict


'''
两层时
building model... ...
_inputs Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
0 _inputs o Tensor("encoder/embedding_lookup/Identity:0", shape=(?, ?, 1024), dtype=float32)
index,output 0 (<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
index:0,state:(LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))
1 _inputs o Tensor("encoder/bidirectional-rnn/concat:0", shape=(?, ?, 2048), dtype=float32)
index,output 1 (<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/transpose_1:0' shape=(?, ?, 1024) dtype=float32>, <tf.Tensor 'encoder/bidirectional-rnn_1/ReverseSequence:0' shape=(?, ?, 1024) dtype=float32>)
index:1,state:(LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))
encoder_outputs Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)
state (LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))
encoder_state (LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 1024) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 1024) dtype=float32>, h=<tf.Tensor 'encoder/bidirectional-rnn_1/bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 1024) dtype=float32>))
decoder encoder_outputs Tensor("encoder/bidirectional-rnn_1/concat:0", shape=(?, ?, 2048), dtype=float32)
Created new model parameters..


'''

'''
bi
Training:   0%|          | 0/222 [00:00<?, ?it/s]/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tqdm/_monitor.py:89: TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481)
  TqdmSynchronisationWarning)
Training:   2%|▏         | 4/222 [01:39<1:30:06, 24.80s/it]----- Step 5 -- Loss 9.66 -- Perplexity 15623.77
Training:   4%|▍         | 9/222 [03:13<1:16:14, 21.48s/it]----- Step 10 -- Loss 9.60 -- Perplexity 14750.59
Training:   6%|▋         | 14/222 [04:47<1:11:12, 20.54s/it]----- Step 15 -- Loss 9.68 -- Perplexity 16053.63
Training:   9%|▊         | 19/222 [06:22<1:08:04, 20.12s/it]----- Step 20 -- Loss 9.13 -- Perplexity 9192.52


no-bi

Training:   0%|          | 0/222 [00:00<?, ?it/s]/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tqdm/_monitor.py:89: TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481)
  TqdmSynchronisationWarning)
Training:   2%|▏         | 4/222 [00:28<25:28,  7.01s/it]----- Step 5 -- Loss 9.67 -- Perplexity 15790.53
Training:   4%|▍         | 9/222 [01:05<26:01,  7.33s/it]----- Step 10 -- Loss 9.66 -- Perplexity 15639.87
Training:   6%|▋         | 14/222 [01:39<24:34,  7.09s/it]----- Step 15 -- Loss 9.35 -- Perplexity 11470.32
Training:   9%|▊         | 19/222 [02:11<23:23,  6.91s/it]----- Step 20 -- Loss 9.01 -- Perplexity 8210.38
Training:  11%|█         | 24/222 [02:38<21:47,  6.60s/it]----- Step 25 -- Loss 9.45 -- Perplexity 12756.64

'''