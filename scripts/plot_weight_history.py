import matplotlib.pyplot as plt
import numpy as np
import os, sys

sum_of_weights_history = None

if __name__ == "__main__":
    weight_history_file = sys.argv[-1]
    with open(weight_history_file, "r") as f:
        show_labels=dict()
        for line in f:
            line = line.split("\t")
            if len(line) == 1 or not line[-1].strip():
                continue
            data = line[-1]
            data = list(map(float, map(str.strip, data.split(","))))
            title = "\t".join(line[:-1]).strip()
            plt.plot(list(range(len(data))), data, label=title)
            show_labels[title] = data[-1]
        plt.show()
        plt.title("Weight history")
        show_labels = sorted(show_labels.keys(), key=lambda x: -show_labels[x])[:7]

    with open(weight_history_file, "r") as f:
        for line in f:
            line = line.split("\t")
            if len(line) == 1 or not line[-1].strip():
                continue
            data = line[-1]
            data = list(map(float, map(str.strip, data.split(","))))
            title = "\t".join(line[:-1])
            if title == "current" or title in show_labels:
                plt.plot(list(range(len(data))), data, label=title)

        plt.legend(loc='best')
        plt.title("Weight history")
        plt.show()