# Normalized Advantage Functions (NAF) in TensorFlow

TensorFlow implementation of [Continuous Deep q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748).

![algorithm](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/algorithm.png)


## Requirements

- Python 2.7
- [gym](https://github.com/openai/gym)
- [TensorFlow](https://www.tensorflow.org/) 0.9+


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for an environment with a continuous action space:

    $ python main.py --env=Pendulum-v0 --is_train=True
    $ python main.py --env=Pendulum-v0 --is_train=True --display=True

To test and record the screens with gym:

    $ python main.py --env=Pendulum-v0 --is_train=False
    $ python main.py --env=Pendulum-v0 --is_train=False --display=True


## Results

Training details of `Pendulum-v0` with different hyperparameters.

    $ python main.py --env=Pendulum-v0 # dark green
    $ python main.py --env=Pendulum-v0 --action_fn=tanh # light green
    $ python main.py --env=Pendulum-v0 --use_batch_norm=True # yellow
    $ python main.py --env=Pendulum-v0 --use_seperate_networks=True # green

![Pendulum-v0_2016-07-15](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/Pendulum-v0_2016-07-15.png)


## References

- [rllab](https://github.com/rllab/rllab.git)
- [keras implementation](https://gym.openai.com/evaluations/eval_CzoNQdPSAm0J3ikTBSTCg)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
