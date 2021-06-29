#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
import matplotlib.pyplot as plt

def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    history = train(cfg, tubs, model, model_type)
    # subplot1 = fig.add_subplot(3, 1, 1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.xlabel("epoch")
    # plt.title('Loss')
    # subplot2 = fig.add_subplot(3, 1, 2)
    # subplot2.plot(history.history['angle_loss'])
    # subplot2.plot(history.history['val_angle_loss'])
    # subplot2.set_title('angle Loss')
    # subplot3 = fig.add_subplot(3, 1, 3)
    # subplot3.plot(history.history['throttle_loss'])
    # subplot3.plot(history.history['val_throttle_loss'])
    # subplot3.set_title('throttle Loss')
    plt.show()

if __name__ == "__main__":
    main()