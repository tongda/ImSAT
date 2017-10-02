import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

START_WORD_INDEX = 1
END_WORD_INDEX = 2
CAPTION_MAX_LENGTH = 16


def create_loss(outputs, captions, length):
  with tf.variable_scope('loss'):
    outputs = tf.Print(outputs, [tf.shape(outputs), tf.shape(captions)], "Outputs & Captions Shape: ")
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

    outputs = tf.Print(outputs, [tf.shape(outputs), tf.shape(captions)], "Outputs & Captions Shape: ")
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=outputs[:, :-1, :], labels=captions[:, 1:])
    losses_length = tf.shape(losses)[1]
    loss_mask = tf.sequence_mask(
      tf.to_int32(length), losses_length)
    losses = losses * tf.to_float(loss_mask)
    loss_size = tf.to_float(tf.reduce_sum(length - 1))
    total_loss = tf.reduce_sum(losses)
    loss = tf.div(total_loss, loss_size)
  return loss


def _batch_norm(x, mode='train', name=None):
  return tf.contrib.layers.batch_norm(inputs=x,
                                      decay=0.95,
                                      center=True,
                                      scale=True,
                                      is_training=(mode == 'train'),
                                      updates_collections=None,
                                      scope=(name + 'batch_norm'))


class AttendTell:
  def __init__(self,
               word_to_idx, dim_feature=(196, 512), dim_embed=512, dim_hidden=1024, n_time_step=16,
               prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
    self.prev2out = prev2out
    self.ctx2out = ctx2out
    self.alpha_c = alpha_c
    self.selector = selector
    self.dropout = dropout
    self.vocab_size = len(word_to_idx)
    self.position_num = dim_feature[0]
    self.feature_length = dim_feature[1]
    self.embedding_size = dim_embed
    self.hidden_size = dim_hidden
    self.T = n_time_step

    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.constant_initializer(0.0)
    self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

  def build(self, features, captions):
    cap_shape = tf.shape(captions)

    batch_size = cap_shape[0]
    bucket_size = cap_shape[1]

    # batch normalize feature vectors
    features = _batch_norm(features, mode='train', name='conv_features')

    c, h, f = self._get_initial_lstm(features=features)
    x = self._word_embedding(inputs=captions)
    # (batch, position_num, feature_size)
    features_proj = self._project_features(features=features)

    # TODO: Try other structures
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)

    def condition(time, all_outputs: tf.TensorArray, caps, states):
      def has_end_word(t):
        return tf.reduce_any(tf.equal(t, END_WORD_INDEX))

      def check_all_ends():
        word_indexes = tf.argmax(all_outputs.stack(), axis=2)
        word_indexes = tf.transpose(word_indexes, [1, 0])
        end_word_flags = tf.map_fn(has_end_word, word_indexes, dtype=tf.bool)
        check_res = tf.reduce_all(end_word_flags)
        return check_res

      all_outputs_size = all_outputs.size()
      is_first_frame = tf.equal(all_outputs_size, 0)
      gen_ends = tf.cond(is_first_frame,
                         lambda: tf.constant(False, tf.bool),
                         check_all_ends)
      cond_res = tf.logical_and(tf.logical_not(gen_ends), tf.less(time, bucket_size))
      return cond_res

    # inputs shape: (batch, embedding)
    # output shape: (batch, hidden_size)
    # hidden layer shape: (embedding, hidden_size)
    # h: (batch, hidden_size)
    def body(time, all_outputs: tf.TensorArray, caps, state: LSTMStateTuple):
      inputs = caps[:, time, :]
      # todo: attention layer missing here

      # context: (batch, feature_size)
      # alpha: (batch, position_num)
      context, alpha = self._attention_layer(features, features_proj, state.h)

      # todo: alpha regularization
      # todo: selector

      # decoder_input: (batch, embedding_size + feature_size)
      decoder_input = tf.concat(values=[inputs, context], axis=1)
      output, nxt_state = lstm_cell(decoder_input, state=state)
      # todo: more complex decode lstm output
      logits = fully_connected(inputs=output,
                               num_outputs=self.vocab_size,
                               activation_fn=None)
      all_outputs = all_outputs.write(time, logits)
      return time + 1, all_outputs, caps, nxt_state

    out_ta = tensor_array_ops.TensorArray(tf.float32,
                                          size=0,
                                          dynamic_size=True,
                                          clear_after_read=False,
                                          element_shape=(None, self.vocab_size))
    init_state = LSTMStateTuple(c, h)

    final_time, outputs, final_inputs, final_state = control_flow_ops.while_loop(
      condition,
      body,
      loop_vars=[0, out_ta, x, init_state]
    )
    outputs = outputs.stack()
    outputs = tf.transpose(outputs, (1, 0, 2))
    return outputs

  def _attention_layer(self, features, features_proj, h):
    with tf.variable_scope('attention_layer'):
      w = tf.get_variable('w', [self.hidden_size, self.feature_length], initializer=self.weight_initializer)
      b = tf.get_variable('b', [self.feature_length], initializer=self.const_initializer)
      w_att = tf.get_variable('w_att', [self.feature_length, 1], initializer=self.weight_initializer)

      # feature_proj is (batch_size, posision_num, feature_size)
      # tf.expand_dims(tf.matmul(h, w), 1) + b) is (batch, 1, feature_size)
      # todo:
      #   why `add` two projected feature here?
      #   I highly suspect we should call `multiply` here
      h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
      # this is just a linear regression
      out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.feature_length]), w_att),
                           [-1, self.position_num])  # (N, L)
      alpha = tf.nn.softmax(out_att)
      context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
      return context, alpha

  def _get_initial_lstm(self, features):
    with tf.variable_scope('initial_lstm'):
      features_mean = tf.reduce_mean(features, 1)

      w_h = tf.get_variable('w_h', [self.feature_length, self.hidden_size], initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.hidden_size], initializer=self.const_initializer)
      h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

      w_c = tf.get_variable('w_c', [self.feature_length, self.hidden_size], initializer=self.weight_initializer)
      b_c = tf.get_variable('b_c', [self.hidden_size], initializer=self.const_initializer)
      c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
      return c, h, features

  def _word_embedding(self, inputs, reuse=False):
    with tf.variable_scope('word_embedding', reuse=reuse):
      w = tf.get_variable('w', [self.vocab_size, self.embedding_size], initializer=self.emb_initializer)
      x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
      return x

  # todo: I think this function has some issue. what is this function used for ?
  def _project_features(self, features):
    with tf.variable_scope('project_features'):
      w = tf.get_variable('w', [self.feature_length, self.feature_length], initializer=self.weight_initializer)
      features_flat = tf.reshape(features, [-1, self.feature_length])
      features_proj = tf.matmul(features_flat, w)
      features_proj = tf.reshape(features_proj, [-1, self.position_num, self.feature_length])
      return features_proj

  def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
    with tf.variable_scope('logits', reuse=reuse):
      w_h = tf.get_variable('w_h', [self.hidden_size, self.embedding_size], initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.embedding_size], initializer=self.const_initializer)
      w_out = tf.get_variable('w_out', [self.embedding_size, self.vocab_size], initializer=self.weight_initializer)
      b_out = tf.get_variable('b_out', [self.vocab_size], initializer=self.const_initializer)

      if dropout:
        h = tf.nn.dropout(h, 0.5)
      h_logits = tf.matmul(h, w_h) + b_h

      if self.ctx2out:
        w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
        h_logits += tf.matmul(context, w_ctx2out)

      if self.prev2out:
        h_logits += x
      h_logits = tf.nn.tanh(h_logits)

      if dropout:
        h_logits = tf.nn.dropout(h_logits, 0.5)
      out_logits = tf.matmul(h_logits, w_out) + b_out
      return out_logits
