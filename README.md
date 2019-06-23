Real valued sequence to sequence autoencoder
---
Most sequence to sequence autoencoders I can find are suitable for categorical sequences,
 such as translation.
 
This auto encoder is for real valued sequences.

The input and output can be multi dimensional, have different dimensions and could even be totally different


### Compatibility
- This package is compatible with both Python 2 and 3
- Example scripts in `main.py` and `main.ipynb` are also compatible with both
- I have no plan to use tensorflow 2.0, so all the nonsense about future obsolescence is muted


### Installation

```bash
pip install seq2seq
```


### Usage
First create a factory:
```python
from models import  NDS2SAEFactory

n_iterations = 3000
factory = NDS2SAEFactory()
factory.set_output('toy.zip')
factory.input_dim = 2 #  Input is 2 dimensiona;
factory.output_dim = 1 #  Output is one dimensional
factory.layer_sizes = [50, 30]

# Use exponential decaying learning rate, in here it starts at 0.02 then decreases exponentially to 0.00001 
factory.lrtype = 'expdecay'
factory.lrargs = dict(start_lr=0.02, finish_lr=0.00001, decay_steps=n_iterations)

# Alternatively, set a constant rate is also possible, e.g. 0.001
# factory.lrtype = 'constant'
# factory.lrargs = dict(lr=0.001)

# For the dropout layer. Default is None. If None, the dropout layer is not used
factory.keep_prob = 0.7

# The hidden layer will be symmetric (in this case: 50:30:30:50)
# otherwise it'll be repeated (50:30:50:30)
factory.symmetric = True

# Save or load (and resume) from this zip file
encoder = factory.build()
```

Create a training sample generator and a validation sample generator. Both should have the same signature:
```python
def generate_samples(batch_size):
    """
    :return in_seq: a list of input sequences. Each sequence must be a np.ndarray
            out_seq: a list of output sequences. Each sequence must be a np.ndarray
                These sequences don't need to be the same length and don't need any padding
                The encoder will take care of that
            last_batch: True if this batch is the last of the iteration.
                E.g. if you have 70000 samples and the batch size is 20000, you'll have 4 batches
                     the last batch contains 10000 samples. You should return False for the first 3
                     batches and True for the last one
    """
    ...
    return in_seq, out_seq, last_batch
```

#### Train

```python
encoder.train(train_generator, valid_generator, n_iterations=3000, batch_size=100, display_step=100)
```

#### Predict

```python
# test_seq is a list of np.ndarrays
predicted = encoder.predict(test_seq)

# predicted is a list of np.ndarrays. Each sequence will have the same length (due to padding)
# Look for the stop token to truncate the padding out
```

#### Encode

```python
# test_seq is a list of np.ndarrays
encoded = encoder.encode(test_seq)

# encoded is a list of hidden-layer states corresponding to each input sequence
```

### Jupyter notebook
Open [main.ipynb](main.ipynb) to run the example

### Licence
MIT

PRs are welcome