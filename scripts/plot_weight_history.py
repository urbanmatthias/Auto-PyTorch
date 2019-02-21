import matplotlib.pyplot as plt
import numpy as np
import os, sys

sum_of_weights_history = None

if __name__ == "__main__":
    weight_history_file = sys.argv[-1]
    with open(weight_history_file, "r") as f:
        data = next(f).split(": ")[1]
        data = list(map(float, map(str.strip, data.split(","))))
        plt.title("Sum of weights history")
        plt.plot(list(range(len(data))), data)
        plt.show()
        next(f)

        show_labels=dict()
        for line in f:
            line = line.split(": ")
            data = line[-1]
            data = list(map(float, map(str.strip, data.split(","))))
            title = ": ".join(line[:-1])
            plt.plot(list(range(len(data))), data, label=title)
            show_labels[title] = np.sum(data)
        plt.ylim((-0.05, 1.05))
        plt.show()
        show_labels = sorted(show_labels.keys(), key=lambda x: -show_labels[x])[:10]

    with open(weight_history_file, "r") as f:
        next(f)
        next(f)

        for line in f:
            line = line.split(": ")
            data = line[-1]
            data = list(map(float, map(str.strip, data.split(","))))
            title = ": ".join(line[:-1])
            if title in show_labels:
                plt.plot(list(range(len(data))), data, label=title)

        plt.legend(loc='best')
        plt.ylim((-0.05, 1.05))
        plt.show()