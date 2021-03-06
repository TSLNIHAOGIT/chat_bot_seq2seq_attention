# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from seq2seq import decoder
from seq2seq import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

import tensorflow as tf

__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
    "CopyNetDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    # if not rnn_cell_impl.assert_like_rnncell(cell):  # pylint: disable=protected-access
    #   raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer.compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  # @property
  # def output_size(self):
  #   # Return the cell output and the id
  #   return BasicDecoderOutput(
  #       rnn_output=self._rnn_output_size(),
  #       # sample_id=tensor_shape.TensorShape([])
  #       sample_id = self._helper.sample_ids_shape))

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)

# from tensorflow.contrib.seq2seq import BahdanauAttention,AttentionWrapper,BasicDecoder
class CopyNetDecoder(BasicDecoder):
    """
    copynet decoder, refer to the paper Jiatao Gu, 2016, 
    'Incorporating Copying Mechanism in Sequence-to-Sequence Learninag'
    https://arxiv.org/abs/1603.06393
    """
    def __init__(self, config, cell, helper, initial_state, 
                                    encoder_outputs, output_layer):
        """Initialize CopyNetDecoder.
        """
        if output_layer is None:
            raise ValueError("output_layer should not be None")
        assert isinstance(helper, helper_py.CopyNetTrainingHelper)
        self.encoder_outputs = encoder_outputs
        encoder_hidden_size = self.encoder_outputs.shape[-1].value

        #权重，需要训练的变量
        self.copy_weight = tf.get_variable('copy_weight', 
                                [encoder_hidden_size, cell.output_size])
        super(CopyNetDecoder, self).__init__(cell, helper, initial_state, 
                                    output_layer=output_layer)
        #自己加的
        self.config=config

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size() + 
                        tf.convert_to_tensor(self.config['encoder_max_seq_len']),
            sample_id=tensor_shape.TensorShape([]))
    
    def shape(self, tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])

    def _mix(self, generate_scores, copy_scores):
        # TODO is this correct? should verify the following code.
        """
        B is batch_size, V is vocab_size, L is length of every input_id
        print genreate_scores.shape     --> (B, V)
        print copy_scores.shape         --> (B, L)
        print self._helper.inputs_ids   --> (B, L)
        """
        # mask is (B, L, V)
        mask = tf.one_hot(self._helper.encoder_inputs_ids, self.config['vocab_size'])
        print('mask',mask)#mask Tensor("decoder/decoder/while/BasicDecoderStep/one_hot:0", shape=(?, ?, 10000), dtype=float32)

        # # choice one, move generate_scores to copy_scores
        # expanded_generate_scores = tf.expand_dims(generate_scores, 1) # (B,1,V)
        # actual_copy_scores = copy_scores + tf.reduce_sum(
        #                         mask * expanded_generate_scores, 2)
        # actual_generate_scores = generate_scores - tf.reduce_sum(
        #                         mask * expanded_generate_scores, 1)

        # choice two, move copy_scores to generate_scores

        expanded_copy_scores = tf.expand_dims(copy_scores, 2)
        print('expanded_copy_scores',expanded_copy_scores)
        #expanded_copy_scores Tensor("decoder/decoder/while/BasicDecoderStep/ExpandDims_1:0", shape=(?, ?, 1), dtype=float32)


        actual_generate_scores = generate_scores + tf.reduce_sum(
                                    mask * expanded_copy_scores, 1)
        print('actual_generate_scores',actual_generate_scores)
        actual_copy_scores = copy_scores - tf.reduce_sum(
                                    mask * expanded_copy_scores, 2)
        print('actual_copy_scores',actual_copy_scores)


        mix_scores = tf.concat([actual_generate_scores, actual_copy_scores], 1)
        print('mix_scores',mix_scores)
        mix_scores = tf.nn.softmax(mix_scores, -1) # mix_scores is (B, V+L)
        print('mix_scores',mix_scores)

        # make sure mix_socres.shape is (B, V + encoder_max_seq_len)
        #
        # padding_size = self.config['encoder_max_seq_len'] - self.shape(copy_scores)[1]
        # mix_scores = tf.pad(mix_scores, [[0, 0], [0, padding_size]])
                
        return mix_scores
    
    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.

        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            print('inputs',inputs)#inputs Tensor("decoder/decoder/while/Identity_14:0", shape=(?, 1024), dtype=float32)

            #这里是AttentionWrapper的输出（使用attention时为）attention, next_state;
            # 与attention_wraper的call调用输入一样call(self, inputs, state):
            # 输出也一样cell_output（attention）, next_state，调用一次call方法就获得当前t的输出，和给下一步t+1用的next_state
            cell_outputs, cell_state = self._cell(inputs, state)
            print('cell_outputs',cell_outputs)#cell_outputs Tensor("decoder/decoder/while/BasicDecoderStep/decoder/Attention_Wrapper/concat_2:0", shape=(?, 1024), dtype=float32)
            #_output_layer只是对cell_outputs（attention）加了全连接层而已，作为输出
            generate_scores = self._output_layer(cell_outputs)
            print('generate_scores',generate_scores)
            #generate_scores Tensor("decoder/decoder/while/BasicDecoderStep/dense/BiasAdd:0", shape=(?, 15829), dtype=float32)

            expand_cell_outputs = tf.expand_dims(cell_outputs, 1)

            # tf.tensordot(a,b,1)a的last 1 axis与b的 first 1 axis 对应元素相乘相加
            copy_scores = tf.tensordot(self.encoder_outputs, self.copy_weight, 1)

            print('self.encoder_outputs',self.encoder_outputs,'\n',
                  'self.copy_weight',self.copy_weight,'\n',
                  'copy_scores',copy_scores)
            copy_scores = tf.nn.tanh(copy_scores)
            copy_scores = tf.reduce_sum(copy_scores * expand_cell_outputs, 2)

            print('copy_scores_shape',copy_scores.shape,copy_scores)

            #词汇10000时，下面这一步已经开始报错
            mix_scores = self._mix(generate_scores, copy_scores)
            print('mix_scores 1',mix_scores)
            #mix_scores 1 Tensor("decoder/decoder/while/BasicDecoderStep/Softmax:0", shape=(?, ?), dtype=float32)

            sample_ids = self._helper.sample(
                time=time, outputs=mix_scores, state=cell_state)
            # sample_ids are not always valid.. TODO
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=mix_scores,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(mix_scores, sample_ids)
        print('outputs',outputs)
        #outputs BasicDecoderOutput(rnn_output=<tf.Tensor 'decoder/decoder/while/BasicDecoderStep/Softmax:0' shape=(?, ?) dtype=float32>, sample_id=<tf.Tensor 'decoder/decoder/while/BasicDecoderStep/TrainingHelperSample/Cast:0' shape=(?,) dtype=int32>)

        return (outputs, next_state, next_inputs, finished)
