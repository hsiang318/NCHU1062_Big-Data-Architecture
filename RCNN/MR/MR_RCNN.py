import tensorflow as tf
import numpy as np
import data_helpers
import collections
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.INFO)

#hyperparameters
conf = collections.namedtuple("conf", "pos, neg, sen_len, test_sam, voc, batch_size, emb_size,"
                              "fil_num, dro_rate, class_num, data_type, context_dim")

conf.pos = "./data/rt-polaritydata/rt-polarity.pos"
conf.neg = "./data/rt-polaritydata/rt-polarity.neg"
conf.sen_len = 56
conf.test_sam = 0.1
conf.voc = 18758
conf.batch_size = 64
conf.emb_size = 50
conf.fil_num = 100
conf.dro_rate = 0.5
conf.class_num = 2
conf.context_dim = 50
conf.test_num = 1066
conf.data_type = tf.float32


# Model
def model_fn_1(features, labels, mode):

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, conf.sen_len])

  ###word embedding
  W = tf.Variable(
      tf.random_uniform([conf.voc, conf.emb_size], -1.0, 1.0),
      dtype=conf.data_type, name="W")
  embedded_chars = tf.nn.embedding_lookup(W, input_layer, None)


  if mode == tf.estimator.ModeKeys.TRAIN :
      input_z = tf.ones([conf.batch_size, conf.context_dim], dtype=conf.data_type)
  else:
      input_z = tf.ones([conf.test_num, conf.context_dim], dtype=conf.data_type)

  # left context
  c_l1 = tf.get_variable("c_l1", shape=[conf.context_dim], dtype=conf.data_type)
  c_l = input_z * c_l1  # c_l = c_l1  -->  配合batch大小差別 : 64/1066，用來更新 c_l(i)
  c_left = tf.reshape(c_l, [-1, 1, conf.context_dim])  # c_left = c_l  -->  為了 tf.concat
  w_l = tf.get_variable("w_l", [conf.context_dim, conf.context_dim], dtype=conf.data_type)
  w_sl = tf.get_variable("w_sl", [conf.emb_size, conf.context_dim], dtype=conf.data_type)

  # right context
  c_rn = tf.get_variable("c_rn", [conf.context_dim], dtype=conf.data_type)
  c_r = input_z * c_rn  # c_r = c_rn  -->  為了變成 tensor
  c_right = tf.reshape(c_r, [-1, 1, conf.context_dim])  # c_left = c_l  -->  為了 tf.concat
  w_r = tf.get_variable("w_r", [conf.context_dim, conf.context_dim], dtype=conf.data_type)
  w_sr = tf.get_variable("w_sr", [conf.emb_size, conf.context_dim], dtype=conf.data_type)

  b = tf.zeros([conf.context_dim], dtype=conf.data_type)
  for i in range(conf.sen_len - 1):
      c_left_c = tf.nn.xw_plus_b(c_l, w_l, b, name="c_left_c")
      c_left_e = tf.nn.xw_plus_b(embedded_chars[:, i, :], w_sl, b, name="c_left_e")
      c_l = tf.nn.relu(c_left_c + c_left_e)  # 迭代計算出 c_l2, c_l3, ..., c_l56
      c_left = tf.concat([c_left, tf.reshape(c_l, [-1, 1, conf.context_dim])], 1)

      c_right_c = tf.nn.xw_plus_b(c_r, w_r, b, name="c_right_c")
      c_right_e = tf.nn.xw_plus_b(embedded_chars[:, conf.sen_len - 1 - i, :], w_sr, b, name="c_right_e")
      c_r = tf.nn.relu(c_right_c + c_right_e)  # 迭代計算出c_l55, c_l54, ..., c_l1
      c_right = tf.concat([tf.reshape(c_r, [-1, 1, conf.context_dim]), c_right], 1)  # 逐一加 c_l55, c_l54, ..., c_l1

  total_length = conf.context_dim + conf.emb_size + conf.context_dim  # total_length = 150
  x_input = tf.concat([c_left, embedded_chars, c_right], -1)

  #每個 word 做 NN
  conv = []
  w_2 = tf.get_variable("w_2", [total_length, conf.fil_num], dtype=conf.data_type)  # w_2 : 150*100
  b_2 = tf.Variable(tf.constant(0.1, shape=[conf.fil_num]), name="b_2")  # b_2 : 100*1

  for i in range(conf.sen_len):
      dense_layer = tf.nn.xw_plus_b(tf.reshape(x_input[:, i, :], [-1, total_length]), w_2, b_2,
                                    name="dense_layer")   # dense_layer : 100*1，NN : 對 xi(每個word)

      conv.append(tf.reshape(tf.nn.tanh(dense_layer),
                             [-1, 1, conf.fil_num]))  # conv.append : 1*100 * 56，逐一加 y1, y2, ..., y56
  conv = tf.concat(conv, 1)  # conv : 56*100

  #max pooling
  pool = tf.nn.max_pool(value=tf.reshape(conv, [-1, 1, conf.sen_len, conf.fil_num]),  # conv : 1*56 *100
                        ksize=[1, 1, conf.sen_len, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID")

  #dropout
  dropout = tf.layers.dropout(
      inputs=pool, rate=conf.dro_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

  #fully connected layer
  W1 = tf.get_variable("W",
      shape=[conf.fil_num, conf.class_num],
      initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.constant(0.1, shape=[conf.class_num]), name="b")

  #scores
  scores = tf.nn.xw_plus_b(tf.reshape(dropout, [-1, conf.fil_num]), W1, b1, name="scores")

  #predictions
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=scores, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(scores, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Calculate Loss (for both TRAIN and EVAL modes)
  loss  = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=scores)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      # optimizer = tf.train.AdamOptimizer(0.001)
      # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  #labels1 = tf.argmax(input=labels, axis=1)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main(unused_argv):

    x_data, y = data_helpers.load_data_and_labels(conf.pos, conf.neg)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_data])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_data)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    test_sample_index = -1 * int(conf.test_sam * float(len(y)))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

    x_train1 = np.asarray(x_train)
    y_train1 = np.asarray(y_train, dtype = np.int32)

    x_test1 = np.asarray(x_test)
    y_test1 = np.asarray(y_test, dtype=np.int32)


    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))


    # Create the Estimator
    text_classifier = tf.estimator.Estimator(model_fn=model_fn_1, model_dir="./models")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_train1},
      y=y_train1,
      batch_size=conf.batch_size,
      num_epochs=None,
      shuffle=True)
    text_classifier.train(
      input_fn=train_input_fn,
      steps=45000,
      hooks=[logging_hook])

  # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test1},
      y=y_test1,
      batch_size=conf.test_num,
      num_epochs=1,
      shuffle=False)
    eval_results = text_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()