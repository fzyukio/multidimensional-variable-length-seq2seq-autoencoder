import numpy as np
from matplotlib import pyplot as plt

from models import NDS2SAEFactory

if __name__ == '__main__':
    max_seq_len = 40
    min_seq_len = 20

    n_inputs = 2
    n_neurons = 100
    n_outputs = 1

    len_x = 1000
    x = np.linspace(0, 150, len_x)
    total_start_points = len_x - max_seq_len

    pad_token = 0

    def generate_samples(time):
        x1 = 2 * np.sin(time)
        x2 = 2 * np.cos(time)
        return x1, x2


    def generate_train_samples(batch_size=10):
        start_time_idx = np.random.choice(list(range(total_start_points)), batch_size)
        sequence_lens = np.random.choice(list(range(min_seq_len, max_seq_len + 1)), batch_size).astype(np.int32)

        time_axis = [x[t:(t + seq_len)] for t, seq_len in zip(start_time_idx, sequence_lens)]
        input = []
        output = []
        for t in time_axis:
            sin, cos = generate_samples(t)
            input.append(np.array([sin, cos]).transpose(1, 0))
            output.append(np.array([sin + cos]).transpose(1, 0))

        return input, output

    def adjust_length(seqs, lens):
        for seg, len in zip(seqs, lens):
            seg[len:] = 0

    def show_test():
        test_seq, test_res = generate_train_samples(batch_size=50)
        predicted = encoder.predict(test_seq)
        
        seq_len = len(test_seq[0])
        pre_len = len(predicted[0])

        # adjust_length(predicted, test_seq_len)

        plt.title("Input sequence, predicted and true output sequences")
        i = plt.plot(list(range(seq_len)), test_seq[0], 'o', label='true input sequence')
        p = plt.plot(list(range(seq_len, seq_len + pre_len)), predicted[0], 'ro', label='predicted outputs')
        t = plt.plot(list(range(seq_len, seq_len + seq_len)), test_res[0], 'co', alpha=0.6, label='true outputs')
        plt.legend(handles=[i[0], p[0], t[0]], loc='upper left')
        plt.show()

    def debug():
        test_seq, test_res = generate_train_samples(batch_size=100)
        encoder.debug(test_seq, test_res)

    def run_encode():
        test_seq, test_res = generate_train_samples(batch_size=1)
        encoded = encoder.encode(test_seq)
        print(encoded)

    # def show_sample():
    #     test_seq, test_res, test_seq_len, test_mask = generate_train_samples(batch_size=1)
    #
    #     adjust_length(test_seq, test_seq_len)
    #     adjust_length(test_res, test_seq_len)
    #
    #     plt.title("Input sequence, predicted and true output sequences")
    #     i = plt.plot(range(max_seq_len), test_seq[0], 'o', label='true input sequence')
    #     t = plt.plot(range(max_seq_len, max_seq_len + max_seq_len), test_res[0], 'co', alpha=0.6, label='true outputs')
    #
    #     plt.legend(handles=[i[0], t[0]], loc='upper left')
    #     plt.show()
    #
    # show_sample()

    factory = NDS2SAEFactory()
    factory.starter_lr = 0.02
    factory.lr_decay_rate = 0.001
    factory.input_dim = n_inputs
    factory.output_dim = n_outputs
    factory.layer_sizes = [50, 30]
    encoder = factory.build('toy.zip')

    # If toy.zip exists, the encoder will continue the training
    # Otherwise it'll train a new model and save to toy.zip every {display_step}
    encoder.train(generate_train_samples, generate_train_samples, n_iterations=3000, batch_size=100, display_step=100)

    # Run this to use the trained autoencoder to encode and decode a randomly generated sequence
    # And display them
    show_test()
    # debug()

    # This will print out the encoded (hidden layers) value
    # run_encode()

    # Not necessary if this is the end of the python program
    encoder.cleanup()
