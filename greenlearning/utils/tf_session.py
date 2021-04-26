from .backend import tf

def open_tf_session():
    """Open a Tensorflow session."""
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)