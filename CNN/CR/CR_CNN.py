import tensorflow as tf
import numpy as np
import data_helpers
import collections
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.INFO)

conf = collections.namedtuple("conf", "pos, neg, sen_len, test_sam, voc, batch_size, emb_size,"
                              "fil_num, dro_rate, class_num")

conf.pos = "./data/custrev.pos"
conf.neg = "./data/custrev.neg"
conf.sen_len = 105
conf.test_sam = 0.1
conf.voc = 5335
conf.batch_size = 34
conf.emb_size = 128
conf.fil_num = 128
conf.dro_rate = 0.5
conf.class_num = 2


# Model
def model_fn_1(features, labels, mode):
  # Input Layer

  input_layer = tf.reshape(features["x"], [-1, conf.sen_len])

  #for i in range(64):
    #a = input_layer[i]
  W = tf.Variable(
      tf.random_uniform([conf.voc, conf.emb_size], -1.0, 1.0),
      name="W")
  embedded_chars = tf.nn.embedding_lookup(W, input_layer, None)
  embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

  pooled_outputs = []

  for i in range(3, 6):
    conv = tf.layers.conv2d(
         inputs=embedded_chars_expanded,
         filters=conf.fil_num,
         kernel_size=[i, conf.emb_size],
         padding="VALID",
         activation=tf.nn.relu)

    pools = tf.layers.max_pooling2d(inputs=conv, pool_size=[conf.sen_len - i + 1, 1], strides=1)
    pool_outputs1 = tf.reshape(pools, [-1, conf.fil_num])
    pooled_outputs.append(pool_outputs1)

  pool2 = tf.concat(pooled_outputs, 1)


  #pool_outputs.append(pool)

  dropout = tf.layers.dropout(
      inputs=pool2, rate=conf.dro_rate, training=mode == tf.estimator.ModeKeys.TRAIN)


  W1 = tf.get_variable(
      "W",
      shape=[3*conf.fil_num, conf.class_num],
      initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.constant(0.1, shape = [conf.class_num]), name="b")

  scores = tf.nn.xw_plus_b(dropout, W1, b1, name="scores")

    #scores_outputs.append(scores)

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
  #loss = tf.reduce_mean(losses)


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
    # TODO: This is very crude, should use cross-validation
    test_sample_index = -1 * int(conf.test_sam * float(len(y)))###
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
      num_epochs=2,
      shuffle=True)
    text_classifier.train(
      input_fn=train_input_fn,
      steps=30000,
      hooks=[logging_hook])

  # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test1},
      y=y_test1,
      num_epochs=1,
      shuffle=False)
    eval_results = text_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()