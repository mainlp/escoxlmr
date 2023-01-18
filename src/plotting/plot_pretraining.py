import json
import matplotlib.pyplot as plt

with open("../xlm-experiment-combined-1e-5-128-30k/trainer_state.json", "r") as f:
    data = json.load(f)
    dev_loss = []
    dev_acc_mlm = []
    dev_acc_erp = []
    dev_step = []
    train_loss = []
    train_step = []
    intervals = [1000, 5000, 10000, 15000,
                 20000, 25000,
                 30000]
    label_intervals = [int(i/1000) for i in intervals]
    for stats in data["log_history"]:
        try:
            if stats.get("eval_loss") and stats.get("step") in intervals:
                dev_loss.append(stats["eval_loss"])
                dev_acc_mlm.append(stats["eval_accuracy"])
                dev_acc_erp.append(stats["eval_accuracy_erp"])
                dev_step.append(stats["step"])
            elif stats.get("step") in intervals:
                train_loss.append(stats["loss"])
                train_step.append(stats["step"])
        except KeyError:
            continue

    fig, ax = plt.subplots(figsize=(5, 2.5), ncols=2)
    ax[0].grid(visible=True, axis="both", which="major", linestyle=":", color="grey")
    ax[1].grid(visible=True, axis="both", which="major", linestyle=":", color="grey")

    x_labels = intervals

    data_labels = ["Train Loss", "Dev. Loss"]
    perf_labels = ["MLM", "ERP"]
    colors = ["steelblue", "lightsalmon", "mediumturquoise", "orangered"]

    ax[0].plot(label_intervals, train_loss, color=colors[0], marker="o", mfc="white", mec=colors[0])
    ax[0].plot(label_intervals, dev_loss, color=colors[1], marker="^", mfc="white", mec=colors[1])

    ax[1].plot(label_intervals, dev_acc_mlm, color=colors[2], marker="o", mfc="white", mec=colors[2])
    ax[1].plot(label_intervals, dev_acc_erp, color=colors[3], marker="^", mfc="white", mec=colors[3])

    # handles, labels = ax[0].get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    ax[0].set_xticks(label_intervals)
    ax[0].set_xticklabels(label_intervals)
    ax[0].set_xlabel("Step (x1000)", alpha=.6)

    ax[1].set_xticks(label_intervals)
    ax[1].set_xticklabels(label_intervals)
    ax[1].set_xlabel("Step (x1000)", alpha=.6)

    # prepare y-axis
    ax[0].set_ylabel("Loss", alpha=.6)
    ax[0].legend(labels=data_labels, prop={'size': 9})
    ax[0].set_title("Loss ESCOLM-R", alpha=.6, fontsize=10)

    ax[1].set_ylabel("Accuracy", alpha=.6)
    ax[1].legend(labels=perf_labels, prop={'size': 9})
    ax[1].set_title("Accuracy Objectives", alpha=.6, fontsize=10)

    fig.tight_layout()
    # plt.show()
    plt.savefig("loss.pdf", format="pdf", dpi=300, bbox_inches="tight")
