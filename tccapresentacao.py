import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.5, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 100, 'Display logs per step.')


def run_training(train_X, train_Y):
    X = tf.placeholder(tf.float32, [m, n])
    Y = tf.placeholder(tf.float32, [m, 1])

    # weights
    W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

    # linear model
    activation = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_sum(tf.square(activation - Y)) / (2*m)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            sess.run(optimizer, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})

            if step % FLAGS.display_step == 0:
                print ("Passo:", (step + 1) , " Cost=", format(sess.run(cost, \
                feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})), "W=", sess.run(W), "b=", sess.run(b))

        print ("Treinamento finalizado!")
        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})
        print ("Resultado Cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        
        # print ("O resultado sugeri que dado a entrada 3,5 o resultado para aproximação será")
        !predict_X = np.array([3, 5], dtype=np.float32).reshape((1, 2))

        # Normaliza o valor de X
        #predict_X = (predict_X - mean) / std

        # Calcula o resultado a partir de X
        #predict_Y = tf.add(tf.matmul(predict_X, W),b)
        #print ("O resultado(Y) =", sess.run(predict_Y))


def read_data(filename, read_from_file = True):
    global m, n

    if read_from_file:
        with open(filename) as fd:
            data_list = fd.read().splitlines()
            
            m = len(data_list) # quantidade de linhas no arquivo
            n = 2 # quantidade de colunas para matrix X

            train_X = np.zeros([m, n], dtype=np.float32)
            train_Y = np.zeros([m, 1], dtype=np.float32)

            for i in range(m):
                datas = data_list[i].split(",")
                for j in range(n):
                    train_X[i][j] = float(datas[j])
                train_Y[i][0] = float(datas[-1])
    else:
        m = 2 # quantidade de linhas da matrix X
        n = 2 # quantidade de colunas da matrix X

        train_X = np.array( [[4,3],[4,2]]).astype('float32')
        train_Y = np.array([[ 7.],[ 8.]]).astype('float32')

    return train_X, train_Y


def feature_normalize(train_X):

    global mean, std
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    return (train_X - mean) / std

import sys

def main(argv):

    # Inicia apartir do arquivo
    train_X, train_Y = read_data("data", True)
    train_X = feature_normalize(train_X)
    run_training(train_X, train_Y)

if __name__ == '__main__':
    tf.app.run()
