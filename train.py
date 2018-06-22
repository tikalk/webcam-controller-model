import math

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras
from ast import literal_eval as make_tuple

flags = tf.app.flags
flags.DEFINE_float("lr", 0.0001, "Learning Rate")
flags.DEFINE_string("units", "((50, 0.2), (40, 0.1))", "Configuration of hidden units in the NN."
                    "Expected: tuple of tuple pairs. Each pair represent one hidden layer and one Dropout layer rate"
                    "For instance: \"((100, 0.2), (50, 0.3))\" will create dense hidden layer of 100 followed by "
                    "dropout layer with rate of 0.2. Afterwards, it will create dense layer of 50 followed by "
                    "dropout layer with rate of 0.3. If you wish to have hidden layer without dropout, specify empty"
                    "second value. Example: \"((100,), (50, 0.3))\"")
flags.DEFINE_integer("epochs", 10, "Number of epochs")
flags.DEFINE_float("batch_frac", 0.3, "The fraction of training examples to consider as batch."
                   "For instance, 0.1 will divide the training to 10 batches")
flags.DEFINE_boolean("draw_plot", False, "Whether to draw a plot at the end")
flags.DEFINE_boolean("export_js", False, "Whether to export to a tenorflow.js model")
FLAGS = flags.FLAGS


def create_model(learning_rate=0.0001, nn_unit_pairs=((100, 0.2),)):
    fc_model = keras.Sequential()

    fc_model.add(keras.layers.Flatten(name="input_flatten", input_shape=(7, 7, 256)))

    for unit_pair in nn_unit_pairs:
        units = unit_pair[0]
        fc_model.add(keras.layers.Dense(name="hidden_dense_%d" % units, units=units,
                                        activation="relu",
                                        kernel_initializer="VarianceScaling",
                                        use_bias=False))
        if len(unit_pair) == 2:
            drop_rate = unit_pair[1]
            if drop_rate == 0.0:
                continue
            fc_model.add(keras.layers.Dropout(name="dropout_%d" % units, rate=drop_rate))

    fc_model.add(keras.layers.Dense(name="output_softmax", units=2,
                                    activation="softmax",
                                    kernel_initializer="VarianceScaling",
                                    use_bias=False))

    fc_model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                     loss="categorical_crossentropy", metrics=['accuracy'])

    return fc_model


def load_data(x_in_path, y_in_path, train_fraction=0.9):
    X = np.loadtxt(x_in_path, delimiter=',')
    print("Total data size: %d" % X.shape[0])
    Y = np.loadtxt(y_in_path, delimiter=',')
    X, Y = shuffle(X, Y)  # Shuffle both together to preserve relation between both
    train_size = math.floor(X.shape[0] * train_fraction)
    x_train = X[0:train_size, :].reshape((train_size, 7, 7, 256))
    y_train = Y[0:train_size, :]
    print("Train X: %s" % str(x_train.shape))
    print("Train Y: %s" % str(y_train.shape))
    x_test = X[train_size:, :].reshape((X.shape[0] - train_size, 7, 7, 256))
    y_test = Y[train_size:, :]
    print("Test X: %s" % str(x_test.shape))
    print("Test Y: %s" % str(y_test.shape))
    return x_train, y_train, x_test, y_test


def train(fc_model: keras.Sequential, x_train, y_train, x_test, y_test, batch_frac=0.2, run_epochs=20):
    batch_size = math.floor(x_train.shape[0] * batch_frac)
    history = fc_model.fit(x=x_train, y=y_train, batch_size=batch_size,
                           epochs=run_epochs, validation_data=(x_test, y_test))
    return history


def export_to_javascript(the_model, tfjs_target_dir='static/spaceinvaders/model'):
    tfjs.converters.save_keras_model(the_model, tfjs_target_dir)


def main(unused_argv):
    # Params
    lr = FLAGS.lr
    units = make_tuple(FLAGS.units)
    epochs = FLAGS.epochs
    batch_fraction = FLAGS.batch_frac

    # Build and train a model
    model = create_model()
    X_train, Y_train, X_test, Y_test = load_data("data/xs.csv", "data/ys.csv")
    train_history = train(model, X_train, Y_train, X_test, Y_test,
                          run_epochs=epochs, batch_frac=batch_fraction)

    if FLAGS.export_js:
        export_to_javascript(model)

    if FLAGS.draw_plot:
        # Visualize training results
        plt.rcParams["figure.figsize"] = (10, 6)
        fig, ax1 = plt.subplots()
        loss, = ax1.plot(train_history.history["loss"], label='Train Loss', color='b')
        val_loss, = ax1.plot(train_history.history["val_loss"], label='Val. Loss', color='cyan')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        acc, = ax2.plot(train_history.history["acc"], label="Train Acc", color='r')
        val_acc, = ax2.plot(train_history.history["val_acc"], label="Val. Acc", color='pink')
        ax2.set_ylabel('Acc', color='r')
        ax2.tick_params('y', colors='r')

        plt.title("Left/Right Model\n units=%s, lr=%f, epochs=%d, batch_frac=%f" % (str(units), lr, epochs, batch_fraction))
        plt.xticks(np.arange(0, epochs + 1, 1.0))
        plt.legend(handles=[loss, val_loss, acc, val_acc], loc='center right')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    tf.app.run()
