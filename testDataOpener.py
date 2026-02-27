import pickle
import gzip

with gzip.open("data/mnist.pkl.gz", "rb") as f:
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")


if __name__ == "__main__":
    print(len(test_data[0]))
    print(len(validation_data[0]))
    print(len(training_data[0]))