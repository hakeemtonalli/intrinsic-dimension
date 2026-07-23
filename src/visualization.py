import numpy as np
import matplotlib.pyplot as plt


def plot_mle(gold_mle, gen_mle):
    # get non-null entries in both
    gold_mle = gold_mle[~np.isnan(gold_mle)]
    gen_mle = gen_mle[~np.isnan(gen_mle)]

    # plot boxplots
    plt.boxplot([gold_mle, gen_mle])
    plt.xlabel("Dataset")
    plt.ylabel("Intrinsic Dimension")
