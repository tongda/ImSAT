import tensorflow as tf

from tensorflow.python.training.session_run_hook import SessionRunHook


class IteratorInitializerHook(SessionRunHook):
  def __init__(self, prefix):
    super(IteratorInitializerHook, self).__init__()
    self.prefix = prefix

  def after_create_session(self, session, coord):
    for initializer in tf.get_collection(self.prefix + "_initializer"):
      session.run(initializer)
