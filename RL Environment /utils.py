# utils.py
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = []
    for i in range(len(scores)):
        avg = sum(scores[max(0, i-100):i+1]) / (i+1 if i < 100 else 100)
        running_avg.append(avg)
    plt.plot(x, running_avg)
    plt.title("Running Average of Rewards")
    plt.savefig(figure_file)
