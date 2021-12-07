import csv
import random
from datetime import datetime

import numpy as np
import pypuf.batch
import pypuf.io
import pypuf.simulation.delay
import tensorflow as tf
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def random_num_with_fix_total(maxValue, num):
    # return an array with a fix sum value
    # maxvalue: sum
    # num：number of integer you want
    a = random.sample(range(1, maxValue), k=num - 1)
    a.append(0)
    a.append(maxValue)
    a = sorted(a)
    b = [a[i] - a[i - 1] for i in range(1, len(a))]
    return b


def check_overlap(loc):
    for i in range(len(loc) - 1):
        if loc[i] - loc[i + 1] == 1:
            if loc[i + 1] != 0:
                loc[i + 1] -= 1
    return loc


class EarlyStopCallback(keras.callbacks.Callback):

    def __init__(self, acc_threshold, patience):
        super().__init__()
        self.accuracy_threshold = acc_threshold
        self.patience = patience
        self.default_patience = patience
        self.previous_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}

        # Stop the training when the validation accuracy reached the threshold accuracy
        if float(logs.get('val_accuracy')) > float(self.accuracy_threshold):
            print(f"\nReached {self.accuracy_threshold:2.2%}% accuracy, so stopping training!\n")
            self.model.stop_training = True

        # Stop the training when the validation acc is not enhancing for consecutive patience value
        if int(logs.get('val_accuracy')) < int(self.previous_accuracy):
            self.patience -= 1
            if not self.patience:
                print('\n*************************************************************************************')
                print('************** Break the training because of early stopping! *************************')
                print('*************************************************************************************\n')
                self.model.stop_training = True
        else:
            # Reset the patience value if the learning enhanced!
            self.patience = self.default_patience
        self.previous_accuracy = logs.get('accuracy')


def initialize_and_tranform_PUF(n: int, k: int, N: int, seed_sim: int, noisiness: float, interface: bool,
                                ghost_bit_len: int, group: int, puf: str):
    print("start generate PUF instance")
    if puf == "xpuf" or puf == "apuf":
        puf = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim, noisiness=noisiness)
    if puf == "ffpuf":
        puf = pypuf.simulation.delay.XORFeedForwardArbiterPUF(n=n, k=k, ff=[(15, 31), (3, 53), (47, 61), (28, 44)],
                                                              seed=seed_sim,
                                                              noisiness=noisiness)
    challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_sim)
    responses = puf.eval(challenges)

    if interface == False:
        print("start generate CRPs")
        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        return challenges, responses
    else:
        print("start generate ghost bits")
        ghost_bits = pypuf.io.random_inputs(n=ghost_bit_len, N=N, seed=seed_sim + 1)

        # random loc array to plus
        if group == 0:
            rng = default_rng()
            loc = rng.choice(n, size=ghost_bit_len, replace=False)
            loc = np.sort(loc)[::-1]
            loc = check_overlap(loc)
            # print(loc)
            for i in range(ghost_bit_len):
                # insert to loc[i]
                challenges = np.insert(challenges, loc[i], ghost_bits[:, i], axis=1)
        else:
            # set up consecutive bits length
            group_len = random_num_with_fix_total(ghost_bit_len, group)
            while 1 in group_len:
                group_len = random_num_with_fix_total(ghost_bit_len, group)
                #     make sure group length larger than 1
            rng = default_rng()
            loc = rng.choice(n, size=group, replace=False)
            loc = np.sort(loc)[::-1]
            loc = check_overlap(loc)
            # print(loc)
            for i in range(group):
                start_point = sum(group_len[0:i])
                for j in range(group_len[i]):
                    # insert one each time to the same loc
                    challenges = np.insert(challenges, loc[i], ghost_bits[:, start_point + j],
                                           axis=1)
        print("start generate CRPs")
        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        print(challenges.shape)
        return challenges, responses


def run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        ghost_bit_len: int, group: int, puf: str) -> dict:
    patience = 5
    epochs = 500
    print('hello')
    challenges, responses = initialize_and_tranform_PUF(n, k, N, seed_sim, noisiness, interface, ghost_bit_len, group,
                                                        puf)

    print(challenges.shape)
    print(responses.shape)

    responses = .5 - .5 * responses
    # 2. build test and training sets
    X_train, X_test, y_train, y_test = train_test_split(challenges, responses, test_size=.1)

    # 3. setup early stopping
    callbacks = EarlyStopCallback(0.98, patience)

    print("start building attack model")
    # 4. build network
    if interface == False:
        ghost_bit_len = 0
    model = tf.keras.Sequential()

    model.add(
        layers.Dense(64, activation='tanh', input_dim=n + ghost_bit_len, kernel_initializer='random_normal'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. train
    started = datetime.now()
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=[callbacks], shuffle=True, validation_split=0.01, verbose=1
    )

    # 6. evaluate result
    results = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

    fields = ['k', 'n', 'N', 'noise', 'training_size', 'test_accuracy', 'test_loss', 'time', 'seed', 'bit_length',
              'group']

    # data rows of csv file
    rows = [
        [k, n, N, noisiness, len(y_train), results[1], results[0], datetime.now() - started, seed_sim, ghost_bit_len,
         group
         ]]
    if puf == "ffpuf":
        if interface == 0:
            with open('ffpuf_without_interface.csv', 'a') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                if seed_sim == 0:
                    write.writerow(fields)
                write.writerows(rows)
        if interface == 1:
            with open('ffpuf_interface_plus.csv', 'a') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                if seed_sim == 0:
                    write.writerow(fields)
                write.writerows(rows)
    else:
        if k == 1:
            if interface == 0:
                with open('apuf_without_interface.csv', 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    if seed_sim == 0:
                        write.writerow(fields)
                    write.writerows(rows)
            if interface == 1:
                with open('apuf_interface_plus.csv', 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    if seed_sim == 0:
                        write.writerow(fields)
                    write.writerows(rows)
        if k == 3:
            if interface == 0:
                with open('3_64xpuf_without_interface.csv', 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    if seed_sim == 0:
                        write.writerow(fields)
                    write.writerows(rows)
            if interface == 1:
                with open('3_64xpuf_interface_plus.csv', 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    if seed_sim == 0:
                        write.writerow(fields)
                    write.writerows(rows)


def main(argv=None):
    seed = [0, 17, 44, 60, 65]  # set the seed here
    interface = True  # set the parameter if the interface is added here
    # parameters for the function run()
    # run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
    # ghost_bit_len: int, group: int, puf: str) -> dict:


    # APUF
    ghost_bit_len1 = [4, 5, 6, 7, 8, 10, 12]
    puf = "apuf"
    for i in seed:
        for j in ghost_bit_len1:
            run(64, 1, 10000000, i, 0.00, 100000, interface, j, 0, puf)

    # 3 XPUF
    puf = "xpuf"
    ghost_bit_len2 = [4, 5, 6, 7, 8]
    for i in seed:
        for j in ghost_bit_len2:
            run(64, 3, 10000000, i, 0.00, 100000, interface, j, 0, puf)

    puf = "ffpuf"
    ghost_bit_lenff = [4, 5, 6, 7, 8]
    for i in seed:
        for j in ghost_bit_lenff:
            run(64, 1, 10000000, i, 0.00, 100000, interface, j, 0, puf)
    #


if __name__ == '__main__':
    main()
