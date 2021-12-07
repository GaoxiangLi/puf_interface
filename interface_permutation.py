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
        if loc[i + 1] - loc[i] == 1:
            if loc[i + 1] != 62:
                loc[i + 1] += 1
        if loc[i + 1] - loc[i] == 1:
            if loc[i + 1] != 62:
                loc[i + 1] += 2
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
                                unsatationary_bit_len: int, hop: int, group: int, puf: str):
    if puf == "xpuf" or puf == "apuf":
        puf = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim, noisiness=noisiness)
    if puf == "ffpuf":
        puf = pypuf.simulation.delay.XORFeedForwardArbiterPUF(n=n, k=k, ff=[(11, 15), (3, 43), (49, 62), (23, 44)],
                                                              seed=seed_sim,
                                                              noisiness=noisiness)

    challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_sim)
    responses = puf.eval(challenges)

    if interface == False:
        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        return challenges, responses
    else:
        print("start permutation bit")
        rng = default_rng()
        if group == 0:
            # random generate numbers(unsatationary bit location) without duplicate
            loc = rng.choice(n, size=unsatationary_bit_len, replace=False)
            loc = np.sort(loc)
            loc = check_overlap(loc)
            # print(loc)

        else:
            # random generate group length(# = group number), sum = unsatationary_bit_len
            group_len = random_num_with_fix_total(unsatationary_bit_len, group)
            while 1 in group_len:
                group_len = random_num_with_fix_total(unsatationary_bit_len, group)
                #     make sure group length larger than 1

            # random generate consecutive unsatationary bit start location without duplicate(not over the last index)
            loc = rng.choice(n - max(group_len) - 1, size=group, replace=False)
            loc = np.sort(loc)
            for i in range(group - 1):
                # check start point of each group for overlapping and fix
                if loc[i + 1] - loc[i] <= group_len[i]:
                    loc[i + 1] = (loc[i] + group_len[i] + 1) % n
            loc = np.sort(loc)
            # add consecutive bit location to array loc
            for i in range(group):
                start_point = sum(group_len[0:i])
                for j in range(1, group_len[i]):
                    loc = np.insert(loc, start_point + 1, (loc[start_point] + j) % n)
            loc = np.sort(loc)

        print(loc)
        for i in range(unsatationary_bit_len - 1):
            tmp = challenges[:, loc[(i + hop) % (unsatationary_bit_len - 1)]]
            challenges[:, loc[(i + hop) % (unsatationary_bit_len - 1)]] = challenges[:, loc[(i)]]
            challenges[:, loc[i % (unsatationary_bit_len - 1)]] = tmp
        challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
        # print(challenges.shape)
        return challenges, responses, loc


# another permutation interface code(pair permutation)
# def initialize_and_tranform_PUF(n: int, k: int, N: int, seed_sim: int, noisiness: float, interface: bool,
#                                  num_pairs: int, puf: str):
#     if puf == "xpuf" or puf == "apuf":
#         puf = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim, noisiness=noisiness)
#     if puf == "ffpuf":
#         puf = pypuf.simulation.delay.XORFeedForwardArbiterPUF(n=n, k=k, ff=[(11, 15), (3, 43), (49, 62), (23, 44)],
#                                                               seed=seed_sim,
#                                                               noisiness=noisiness)
#
#     challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_sim)
#     responses = puf.eval(challenges)
#
#     if interface == False:
#         challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
#         return challenges, responses
#     else:
#         print("start permutation bit")
#         rng = default_rng()
#         # random generate numbers(unsatationary bit location) without duplicate
#         loc = rng.choice(n - 2, size=num_pairs, replace=False)
#         loc = np.sort(loc)
#         # check overlap
#         loc = check_overlap(loc)
#
#         # swap loc[i] and loc[i]+1 in challenges[][]
#         for i in range(num_pairs):
#             tmp = challenges[:, loc[i]]
#             challenges[:, loc[i]] = challenges[:, loc[i] + 1]
#             challenges[:, loc[i] + 1] = tmp
#         # transform new challenges
#         challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
#         print(challenges.shape)
#
#         # add consecutive bit location to array loc
#         for i in range(num_pairs):
#             loc = np.insert(loc, len(loc), loc[i] + 1)
#         loc = np.sort(loc)
#         print(loc)
#
#         return challenges, responses, loc


def run(n: int, k: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, interface: bool,
        unsatationary_bit_len: int, hop: int, group: int, puf: str) -> dict:
    patience = 5
    epochs = 500
    print('hello')
    challenges, responses, loc = initialize_and_tranform_PUF(n, k, N, seed_sim, noisiness, interface,
                                                             unsatationary_bit_len, hop, group,
                                                             puf)

    print(challenges.shape)
    print(responses.shape)

    responses = .5 - .5 * responses
    # 2. build test and training sets
    X_train, X_test, y_train, y_test = train_test_split(challenges, responses, test_size=.15)

    # 3. setup early stopping
    callbacks = EarlyStopCallback(0.92, patience)

    # 4. build network
    if interface == False:
        unsatationary_bit_len = 0
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(64, activation='tanh', input_dim=n,
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
    model_loss = history.history["val_loss"]

    epoch_chosen = model_loss.index(min(model_loss)) + 1
    print(epoch_chosen)

    fields = ['k', 'n', 'N', 'noise', 'training_size', 'test_accuracy', 'test_loss', 'time', 'seed',
              "unsatationary_bit_len", 'hop', 'group', 'location', 'stop_epoch']

    # data rows of csv file
    rows = [[k, n, N, noisiness, len(y_train), results[1], results[0], datetime.now() - started, seed_sim,
             unsatationary_bit_len, hop, group, loc, epoch_chosen
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
            with open('ffpuf_permutation_interface.csv', 'a') as f:
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
                with open('apuf_permutation_interface.csv', 'a') as f:
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
                with open('3_64xpuf_permutation_pair.csv', 'a') as f:
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
    #  unsatationary_bit_len: int, hop: int, group: int, puf: str) -> dict:

    # APUF
    puf = "apuf"
    unsatationary_bit_len = [4, 5, 6, 7, 8]
    hop = 2
    noise = 0
    group = 0
    for i in seed:
        for j in unsatationary_bit_len:
            run(64, 1, 10000000, i, noise, 100000, interface, j, hop, group, puf)

    hop = 3
    for i in seed:
        for j in unsatationary_bit_len:
            run(64, 1, 10000000, i, noise, 100000, interface, j, hop, 0, puf)
    # 3 XPUF
    puf = "xpuf"
    unsatationary_bit_len2 = [3, 4, 5]
    hop = 1
    group = 0
    for i in seed:
        for j in unsatationary_bit_len2:
            run(64, 3, 10000000, i, noise, 100000, interface, j, hop, group, puf)

    puf = "ffpuf"
    unsatationary_bit_len2 = [10, 11, 12]
    hop = 3
    group = 0
    for i in seed:
        for j in unsatationary_bit_len2:
            run(64, 1, 10000000, i, 0, 100000, interface, j, hop, group, puf)


if __name__ == '__main__':
    main()
