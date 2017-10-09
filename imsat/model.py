import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, RNNCell

NULL_WORD_INDEX = 0
START_WORD_INDEX = 1
END_WORD_INDEX = 2
CAPTION_MAX_LENGTH = 41


def create_loss(outputs, captions, length):
  with tf.variable_scope('loss'):
    outputs = _align_text(captions, outputs)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=outputs[:, :-1, :], labels=captions[:, 1:])
    loss = _mask_loss(length, losses)

    return loss


def _mask_loss(length, losses):
  """
  Mask losses with the length tensor.
  :param length: The length tensor which refer to the length
    of captions.
  :param losses: cross entropy of between outputs and captions
  :return: summed of masked losses
  """
  losses_length = tf.shape(losses)[1]
  loss_mask = tf.sequence_mask(
    tf.to_int32(length), losses_length)
  losses = losses * tf.to_float(loss_mask)
  loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(length - 1))
  return loss


def _align_text(captions, outputs):
  """
  Mad outputs to make them same length to captions
  :param captions: captions to be aligned.
  :param outputs: outputs generated by decoder
  :return: padded outputs
  """
  bucket_size = tf.shape(captions)[1]
  output_len = tf.shape(outputs)[1]

  # 1st dimension of indices is the No. of dimension to update,
  # since the output shape is (batch, time, word), we should
  # pad on the second dimension, which is `time`.
  indexes = [[1, 1]]
  values = tf.expand_dims(bucket_size - output_len, axis=0)
  # because rank of final outputs tensor is 3, so the shape is (3, 2)
  # for example, output shape is (2, 4, 1000), and caption shape is
  # (2, 6), we should pad the second dimension to 6. So the padding
  # matrix should be ((0, 0), (0, 2), (0, 0))
  shape = [3, 2]
  paddings = tf.scatter_nd(indexes, values, shape)
  outputs = tf.pad(outputs, paddings)
  return outputs


def _batch_norm(x, mode='train', name=None):
  return tf.contrib.layers.batch_norm(inputs=x,
                                      decay=0.95,
                                      center=True,
                                      scale=True,
                                      is_training=(mode == 'train'),
                                      updates_collections=None,
                                      scope=(name + 'batch_norm'))


def _get_cond_fn(max_length=CAPTION_MAX_LENGTH):
  """
  Generate condition function used for tf while loop.
  :param max_length: `Tensor` of `int` value. mark the max step
    when generating captions.
  :return:
    Condition function.
  """

  def condition(time, all_outputs: tf.TensorArray, caps, states):
    def has_end_word(t):
      return tf.reduce_any(tf.equal(t, END_WORD_INDEX))

    def check_all_ends():
      word_indexes = tf.argmax(all_outputs.stack(), axis=2)
      word_indexes = tf.transpose(word_indexes, [1, 0])
      end_word_flags = tf.map_fn(has_end_word, word_indexes, dtype=tf.bool)
      check_res = tf.reduce_all(end_word_flags)
      return check_res

    with tf.variable_scope("cond_fn"):
      all_outputs_size = all_outputs.size()
      is_first_frame = tf.equal(all_outputs_size, 0)
      gen_ends = tf.cond(is_first_frame,
                         lambda: tf.constant(False, tf.bool),
                         check_all_ends)
      cond_res = tf.logical_and(tf.logical_not(gen_ends),
                                tf.less(time, max_length))
    return cond_res

  return condition


class AttendTell:
  def __init__(self,
               vocab_size,
               dim_feature=(196, 512),
               dim_embed=512,
               dim_hidden=1024,
               prev2out=True,
               ctx2out=True,
               alpha_c=0.0,
               selector=True,
               dropout=True,
               hard_attention=True):
    self.sample_method = "multinormial"
    self.keep_prob = 0.5
    self.prev2out = prev2out
    self.ctx2out = ctx2out
    self.alpha_c = alpha_c
    self.selector = selector
    self.dropout = dropout
    self.vocab_size = vocab_size
    self.position_num = dim_feature[0]
    self.feature_length = dim_feature[1]
    self.embedding_size = dim_embed
    self.hidden_size = dim_hidden

    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.constant_initializer(0.0)
    self.emb_initializer = tf.random_uniform_initializer(minval=-1.0,
                                                         maxval=1.0)
    self.hard_attention = hard_attention

  def _get_body_fn(self,
                   rnn_cell: RNNCell,
                   features,
                   dropout=False,
                   use_generated_inputs=False):
    """
    Generate body function used for tf while loop.
    :param rnn_cell:
    :param features:
    :param use_generated_inputs: mark whether to use generated caption as
      inputs for next step.
      If `use_generated_inputs` is false, it means the inputs are all the
      captions, so in the `body_fn`, inputs is extracted by time slice.
      If `use_generated_inputs` is True, it means the inputs are generated
      in the previous step, so it will be used directly.
    :return:
      Body function.
    """

    # inputs shape: (batch, embedding)
    # output shape: (batch, hidden_size)
    # hidden layer shape: (embedding, hidden_size)
    # h: (batch, hidden_size)
    def body_fn(time, all_outputs: tf.TensorArray, inputs, state: LSTMStateTuple):
      with tf.variable_scope("body_fn"):
        if not use_generated_inputs:
          next_inputs = inputs
          inputs = inputs[:, time, :]

        # context: (batch, feature_size)
        # alpha: (batch, position_num)
        context, alpha = self._attention_layer(features, state.h)

        # todo: alpha regularization

        if self.selector:
          with tf.variable_scope("selector"):
            beta = fully_connected(state.h,
                                   num_outputs=1,
                                   activation_fn=tf.nn.sigmoid,
                                   weights_initializer=self.weight_initializer,
                                   biases_initializer=self.const_initializer)
            context = tf.multiply(beta, context, name="selected_context")

        # decoder_input: (batch, embedding_size + feature_size)
        decoder_input = tf.concat(values=[inputs, context], axis=1, name="decoder_input")
        output, nxt_state = rnn_cell(decoder_input, state=state)
        logits = self._decode_rnn_outputs(output, context, inputs, dropout=dropout)
        all_outputs = all_outputs.write(time, logits)
        if use_generated_inputs:
          next_inputs = self._word_embedding(self._sampler(logits), reuse=True)
      return time + 1, all_outputs, next_inputs, nxt_state

    return body_fn

  def _sampler(self, logits):
    if self.sample_method == "argmax":
      return tf.argmax(logits, axis=-1)
    elif self.sample_method == "multinormial":
      return tf.reshape(tf.multinomial(logits, num_samples=1),
                        shape=(-1,))
    else:
      raise Exception("Unknown sample method %s" % self.sample_method)

  def _decode_rnn_outputs(self, output, features, previous, dropout=False):
    with tf.variable_scope("decode_rnn_output"):
      if dropout:
        output = tf.nn.dropout(output, keep_prob=self.keep_prob)
      hidden = fully_connected(inputs=output,
                               num_outputs=self.embedding_size,
                               activation_fn=None)

      if self.ctx2out:
        # ctx2out
        context_hidden = fully_connected(features,
                                         num_outputs=self.embedding_size,
                                         activation_fn=None,
                                         biases_initializer=None)
        hidden += context_hidden

      if self.prev2out:
        # prev2out
        hidden += previous

      hidden = tf.nn.tanh(hidden)

      if dropout:
        hidden = tf.nn.dropout(hidden, keep_prob=self.keep_prob)
      logits = fully_connected(inputs=hidden,
                               num_outputs=self.vocab_size,
                               activation_fn=None)
    return logits

  def _get_rnn_cell(self):
    return tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)

  def build_train(self, features, captions, use_generated_inputs=False):
    with tf.variable_scope("attend_and_tell") as root_scope:
      self.root_scope = root_scope
      cap_shape = tf.shape(captions)
      bucket_size = cap_shape[1]

      features = _batch_norm(features, mode='train', name='conv_features')

      lstm_cell = self._get_rnn_cell()
      cond_fn = _get_cond_fn(bucket_size)

      body_fn = self._get_body_fn(lstm_cell, features, dropout=self.dropout,
                                  use_generated_inputs=use_generated_inputs)
      out_ta = self._get_init_outputs_array()
      init_state = self._get_init_state(features=features)
      if use_generated_inputs:
        loop_vars = [0, out_ta, self._word_embedding(inputs=captions[:, 0]), init_state]
      else:
        embedded_captions = self._word_embedding(inputs=captions)
        loop_vars = [0, out_ta, embedded_captions, init_state]
      _, outputs, _, _ = control_flow_ops.while_loop(cond_fn,
                                                     body_fn,
                                                     loop_vars=loop_vars)
      outputs = tf.transpose(outputs.stack(), (1, 0, 2))
    return outputs

  def build_infer(self, features):
    with tf.variable_scope("attend_and_tell") as root_scope:
      self.root_scope = root_scope
      features = _batch_norm(features, mode='train', name='conv_features')

      lstm_cell = self._get_rnn_cell()
      cond_fn = _get_cond_fn()
      body_fn = self._get_body_fn(lstm_cell, features,
                                  use_generated_inputs=True)
      out_ta = self._get_init_outputs_array()
      zero_inputs = self._zero_inputs(features)
      init_state = self._get_init_state(features=features)
      loop_vars = [0, out_ta, zero_inputs, init_state]
      _, outputs, _, _ = control_flow_ops.while_loop(cond_fn,
                                                     body_fn,
                                                     loop_vars=loop_vars)
      outputs = tf.transpose(outputs.stack(), (1, 0, 2))
    return outputs

  def _get_init_outputs_array(self):
    return tensor_array_ops.TensorArray(tf.float32,
                                        size=0,
                                        dynamic_size=True,
                                        clear_after_read=False,
                                        element_shape=(None, self.vocab_size))

  def _zero_inputs(self, features):
    zero_inputs = tf.fill(tf.expand_dims(tf.shape(features)[0], 0),
                          START_WORD_INDEX)
    zero_inputs = self._word_embedding(zero_inputs)
    return zero_inputs

  def _attention_layer(self, features, h):
    with tf.variable_scope('attention_layer'):
      # (batch, position_num, feature_size)
      features_proj = self._project_features(features=features)

      state_proj = fully_connected(inputs=h,
                                   num_outputs=self.feature_length,
                                   activation_fn=None,
                                   weights_initializer=self.weight_initializer,
                                   biases_initializer=self.const_initializer)
      # todo:
      #   why `add` two projected feature here?
      #   I highly suspect we should call `multiply` here
      # feature_proj is (batch_size, posision_num, feature_size)
      # tf.expand_dims(tf.matmul(h, w), 1) + b) is (batch, 1, feature_size)
      h_att = tf.nn.relu(features_proj + tf.expand_dims(state_proj, 1))

      # this is just a linear regression
      flat_h_att = tf.reshape(h_att, [-1, self.feature_length])
      flat_att = fully_connected(inputs=flat_h_att,
                                 num_outputs=1,
                                 activation_fn=None,
                                 weights_initializer=self.weight_initializer,
                                 biases_initializer=None)
      out_att = tf.reshape(flat_att, [-1, self.position_num])  # (N, L)
      alpha = tf.nn.softmax(out_att)
      # context: (batch, feature_length)
      if self.hard_attention:
        batch_size = tf.shape(features)[0]
        # todo: generate mask for each sample
        sample_mask = tf.to_float(tf.multinomial(tf.fill([batch_size, 2], 0.5), 1)[0][0])
        hard_att = tf.reshape(tf.one_hot(tf.multinomial(alpha, 1), depth=self.position_num),
                              shape=(-1, self.position_num))
        alpha = sample_mask * hard_att + \
                (1 - sample_mask) * alpha
        context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), axis=1, name='context')

      else:
        context = tf.reduce_sum(features * tf.expand_dims(alpha, 2),
                                axis=1,
                                name='context')
      return context, alpha

  def _get_init_state(self, features):
    with tf.variable_scope('initial_lstm'):
      features_mean = tf.reduce_mean(features, 1)

      h = fully_connected(inputs=features_mean,
                          num_outputs=self.hidden_size,
                          activation_fn=tf.nn.tanh,
                          weights_initializer=self.weight_initializer,
                          biases_initializer=self.const_initializer)

      c = fully_connected(inputs=features_mean,
                          num_outputs=self.hidden_size,
                          activation_fn=tf.nn.tanh,
                          weights_initializer=self.weight_initializer,
                          biases_initializer=self.const_initializer)
      return LSTMStateTuple(c, h)

  def _word_embedding(self, inputs, reuse=False):
    with tf.variable_scope(self.root_scope):
      with tf.variable_scope('word_embedding', reuse=reuse), tf.device("/cpu:0"):
        w = tf.get_variable('w',
                            [self.vocab_size, self.embedding_size],
                            initializer=self.emb_initializer)
        x = tf.nn.embedding_lookup(w, inputs, name='word_vector')
      return x

  # todo: I think this function has some issue. what is this function
  #   used for ?

  def _project_features(self, features):
    with tf.variable_scope('project_features'):
      features_flat = tf.reshape(features, [-1, self.feature_length])
      features_proj = fully_connected(
        inputs=features_flat,
        num_outputs=self.feature_length,
        activation_fn=None,
        weights_initializer=self.weight_initializer,
        biases_initializer=None
      )
      features_proj = tf.reshape(features_proj,
                                 [-1, self.position_num, self.feature_length])
      return features_proj
