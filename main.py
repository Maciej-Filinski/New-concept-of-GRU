from NewGRU import NewGRU
import tensorflow as tf


def main():
    model = NewGRU(input_shape=(1, 20), state_length=10)
    x = tf.ones(shape=(1, 10))
    model(x)


if __name__ == "__main__":
    main()
