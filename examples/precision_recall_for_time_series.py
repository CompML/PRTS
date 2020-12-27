import sys
sys.path.append("../prts/")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

from prts import ts_precision, ts_recall

sns.set_style("darkgrid")


def main():
    with open("data//lstm_ad.real", "r") as f:
        real = f.readlines()
    with open("data/lstm_ad.pred", "r") as f:
        pred = f.readlines()

    real = np.array([int(value.strip("\n")) for value in real])
    pred = np.array([int(value.strip("\n")) for value in pred])

    # calculate classic precision and recall score
    precision_classic = precision_score(real, pred)
    recall_classic = recall_score(real, pred)
    print("precision_classic=", precision_classic)
    print("recall_classic=", recall_classic)

    # calculate time series precision score
    precision_flat = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="flat")
    precision_front = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="front")
    precision_middle = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="middle")
    precision_back = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="back")
    print("precision_flat=", precision_flat)
    print("precision_front=", precision_front)
    print("precision_middle=", precision_middle)
    print("precision_back=", precision_back)

    # calculate time series recall score
    recall_flat = ts_recall(real, pred, alpha=0.0, cardinality="reciprocal", bias="flat")
    recall_front = ts_recall(real, pred, alpha=0.0, cardinality="reciprocal", bias="front")
    recall_middle = ts_recall(real, pred, alpha=0.0, cardinality="reciprocal", bias="middle")
    recall_back = ts_recall(real, pred, alpha=0.0, cardinality="reciprocal", bias="back")
    print("recall_flat=", recall_flat)
    print("recall_front=", recall_front)
    print("recall_middle=", recall_middle)
    print("recall_back=", recall_back)

    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    ax1.set_title("Preds and Ground Truth")
    ax1.plot(real*1.25, alpha=0.6)
    ax1.plot(pred, alpha=0.6)
    ax1.legend(["real anomalies", "predicted anomalies"], loc="upper left", fontsize=8)
    ax1.grid(True)

    ax2.set_title("Precision")
    ax2.bar(["precision_classic", "precision_flat", "precision_front", "precision_middle", "precision_back"],
            [precision_classic, precision_flat, precision_front, precision_middle, precision_back], alpha=0.6)
    ax2.set_ylim([0, 1])
    ax2.grid(True)
    ax2.tick_params("x", rotation=45)

    ax3.set_title("Recall")
    ax3.bar(["recall_classic", "recall_flat", "recall_front", "recall_middle", "recall_back"],
            [recall_classic, recall_flat, recall_front, recall_middle, recall_back], alpha=0.6)
    ax3.set_ylim([0, 1])
    ax3.grid(True)

    ax3.tick_params("x", rotation=45)
    plt.show()
    fig.clear()


if __name__ == "__main__":
    main()
