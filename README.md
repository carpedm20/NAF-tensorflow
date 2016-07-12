# Normalized Advantage Functions (NAF) in TensorFlow

TensorFlow implementation of [Continuous Deep q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748).

![algorithm](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/algorithm.png)


## Requirements

- Python 2.7
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
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

Training details of Pendulum-v0 for different parameters. (in progress)

    $ python main.py --env=Pendulum-v0 --learning_rate=0.001 --hidden_dims='[100,100]' # light green
    $ python main.py --env=Pendulum-v0 --learning_rate=0.001 --hidden_dims='[100,100]' --use_batch_norm=True # purple
    $ python main.py --env=Pendulum-v0 --learning_rate=0.001 --hidden_dims='[100,100]' --hidden_activation_fn=relu # yellow
    $ python main.py --env=Pendulum-v0 --learning_rate=0.0001 --hidden_dims='[200,200]' # red
    $ python main.py --env=Pendulum-v0 --learning_rate=0.0001 --hidden_dims='[200,200]' --use_batch_norm=True # orange

![Pendulum-v0_2016-07-12](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/Pendulum-v0_2016-07-12.png)


## References

- [rllab](https://github.com/rllab/rllab.git)
- [keras implementation](https://gym.openai.com/evaluations/eval_CzoNQdPSAm0J3ikTBSTCg)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
