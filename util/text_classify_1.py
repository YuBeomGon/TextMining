import os
import time
import datetime
import absl.app
#from tensorflow import flags
import tensorflow as tf
import numpy as np
import util.text as tool
from util.flags import create_flags, FLAGS

# data loading
data_path = '/home/beomgon2/medical_chart/movie/nsmc/train.csv'
contents, points = tool.loading_rdata(data_path, eng=True, num=True, punc=False)
contents = tool.cut(contents,cut=2)

# tranform document to vector
max_document_length = 50
x, vocabulary, vocab_size = tool.make_input(contents,max_document_length)
print('사전단어수 : %s' % (vocab_size))
y = tool.make_output(points,threshold=2.5)

# divide dataset into train/test set
x_train, x_test, y_train, y_test = tool.divide(x,y,train_prop=0.8)



#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS
import sys
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 3. train the model and test
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        print("debug: ", x_train.shape[1],y_train.shape[1], vocab_size, FLAGS.embedding_dim,
                list(map(int, FLAGS.filter_sizes.split(","))), FLAGS.num_filters)
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=vocab_size,
                      embedding_size=FLAGS.embedding_dim,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]


        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        testpoint = 0
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                if testpoint + 100 < len(x_test):
                    testpoint += 100
                else:
                    testpoint = 0
                print("\nEvaluation:")
                dev_step(x_test[testpoint:testpoint+100], y_test[testpoint:testpoint+100], writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def main(_):
    initialize_globals()

    if FLAGS.train_files:
        tf.reset_default_graph()
        train()

#    if FLAGS.test_files:
#        tf.reset_default_graph()
#        test()

#    if FLAGS.one_shot_infer:
#
#        tf.reset_default_graph()
#        train()


if __nain__ == '__main__':                
    create_flags()
    absl.app.run(main)
