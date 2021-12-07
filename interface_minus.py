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
        if loc[i] - loc[i + 1] <= 0:
            if loc[i + 1] >= 2:
                loc[i + 1] = loc[i] - 2
                if loc[i + 1] < 0:
                    loc[i + 1] = 0
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
                                double_use_bit_len: int, group: int, puf: str):
    if puf == "xpuf" or puf == "apuf":
        puf = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim, noisiness=noisiness)
    if puf == "ffpuf":
        puf = pypuf.simulation.delay.XORFeedForwardArbiterPUF(n=n, k=k, ff=[(15, 31), (3, 53), (47, 61), (28, 44)],
                                                              seed=seed_sim,
                                                              noisiness=noisiness)

    challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_sim)
    responses = puf.eval(challenges)
    if interface == False:
        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        return challenges, responses
    else:
        print("start delete double used bit")
        if group == 0:
            rng = default_rng()
            loc = rng.choice(n - 1, size=double_use_bit_len, replace=False)
            loc = np.sort(loc)[::-1]
            loc = check_overlap(loc)
            print(loc)
            for i in range(double_use_bit_len):
                # delete at loc[i] position
                challenges = np.delete(challenges, loc[i], axis=1)
        else:
            # set up consecutive bits length
            group_len = random_num_with_fix_total(double_use_bit_len, group)
            while 1 in group_len:
                group_len = random_num_with_fix_total(double_use_bit_len, group)
            #     make sure group length larger than 1
            rng = default_rng()
            loc = rng.choice(n - 1, size=group, replace=False)
            loc = np.sort(loc)[::-1]
            loc = check_overlap(loc)
            # print(loc)
            for i in range(group):
                for j in range(group_len[i]):
                    # delete one each time to the same loc
                    if loc[i] >= len(challenges[0]):
                        loc[i] = len(challenges[0]) - 1
                    challenges = np.delete(challenges, loc[i], axis=1)

        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        print(challenges.shape)
        return challenges, responses, loc


def run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        double_use_bit_len: int, group: int, puf: str) -> dict:
    patience = 3
    epochs = 500
    print('hello')
    challenges, responses, loc = initialize_and_tranform_PUF(n, k, N, seed_sim, noisiness, interface,
                                                             double_use_bit_len,
                                                             group, puf)

    print(challenges.shape)
    print(responses.shape)

    responses = .5 - .5 * responses
    # 2. build test and training sets
    X_train, X_test, y_train, y_test = train_test_split(challenges, responses, test_size=.15)

    # 3. setup early stopping
    callbacks = EarlyStopCallback(0.95, patience)


    # 4. build network
    if interface == False:
        double_use_bit_len = 0
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(64, activation='tanh', input_dim=n - double_use_bit_len,
                     kernel_initializer='random_normal'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. train
    started = datetime.now()
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=[callbacks], shuffle=True, validation_split=0.05, verbose=0
    )

    # 6. evaluate result
    results = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

    fields = ['k', 'n', 'N', 'noise', 'training_size', 'test_accuracy', 'test_loss', 'time', 'seed', 'bit_length',
              'group', 'location']

    # data rows of csv file
    rows = [
        [k, n, N, noisiness, len(y_train), results[1], results[0], datetime.now() - started, seed_sim,
         double_use_bit_len,
         group, loc
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
            with open('ffpuf_interface_minus_new.csv', 'a') as f:
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
                with open('apuf_interface_minu.csv', 'a') as f:
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
                with open('3_64xpuf_interface_minu.csv', 'a') as f:
                    # using csv.writer method from CSV package
                    write = csv.writer(f)
                    if seed_sim == 0:
                        write.writerow(fields)
                    write.writerows(rows)


def main(argv=None):
    seed = [0, 6, 17, 44, 60, 231, 634, 232, 654, 543] # set the seed here
    interface = True  # set the parameter if the interface is added here
    # parameters for the function run()
    # run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
    # double_use_bit_len: int, group: int, puf: str) -> dict:

    # APUF
    puf = "apuf"
    double_use_bit_len1 = [1, 2, 3, 4, 5, 6, 7, 8]
    group = 0
    for i in seed:
        for j in double_use_bit_len1:
            run(64, 3, 10000000, i, 0.00, 100000, interface, j, group, puf)


    # 3 XPUF
    puf = "xpuf"
    double_use_bit_len2 = [1, 2, 3, 4, 5, 6, 7]
    group = 0
    for i in seed:
        for j in double_use_bit_len2:
            run(64, 3, 10000000, i, 0.00, 100000, interface, j, group, puf)

    # FFPUF
    puf = "ffpuf"
    double_use_bit_len2 = [1, 2, 3, 4, 5, 6, 7]
    group = 0
    for i in seed:
        for j in double_use_bit_len2:
            run(64, 1, 10000000, i, 0.00, 100000, interface, j, group, puf)
    #


if __name__ == '__main__':
    main()
