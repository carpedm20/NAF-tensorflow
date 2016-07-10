# Normalized Advantage Functions (NAF) in TensorFlow

TensorFlow implementation of [Continuous Deep q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748).

![algorithm](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/algorithm.png)


## Requirements

- Python 2.7
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow](https://www.tensorflow.org/)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for Breakout:

    $ python main.py --env=BipedalWalker-v2 --is_train=True
    $ python main.py --env=BipedalWalker-v2 --is_train=True --display=True

To test and record the screens with gym:

    $ python main.py --env=BipedalWalker-v2 --is_train=False
    $ python main.py --env=BipedalWalker-v2 --is_train=False --display=True


## Results

(in progress)


## References

- [rllab](https://github.com/rllab/rllab.git)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
