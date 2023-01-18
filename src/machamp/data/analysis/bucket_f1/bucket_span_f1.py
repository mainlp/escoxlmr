import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import sys

# precalculated because lazy
xlmr_skillspan = [52.9, 56.8, 60.2, 50.0, 34.73, 52.9]
escoxlmr_skillspan = [58.8, 59.47, 63.18, 52.25, 32.23, 58.80]
support_skillspan = [201, 155, 112, 45, 16]

xlmr_sayfullina = [90.8, 81.3, 57.6, 0.0, 0.0, 90.8]
escoxlmr_sayfullina = [92.9, 85.75, 67.75, 0.0, 0.0, 92.9]
support_sayfullina = [1473, 330, 35, 0, 0]

xlmr_green = [51.05, 54.0, 41.4, 32.2, 39.5, 51.05]
escoxlmr_green = [53.9, 53.85, 45.4, 31.4, 40.35, 53.9]
support_green = [554, 203, 84, 24, 18]

xlmr_jobstack = [78.9, 78.15, 58.25, 63.35, 34.0, 78.9]
escoxlmr_jobstack = [80.85, 77.6, 61.55, 55.0, 29.35, 80.85]
support_jobstack = [278, 74, 8, 3, 2]

xlmr_gnehm = [88.95, 71.80, 72.55, 0.0, 43.35, 88.95]
escoxlmr_gnehm = [90.1, 77.7, 72.2, 0.0, 30.0, 90.1]
support_gnehm = [979, 64, 22, 3, 2]

xlmr_fijo = [47.05, 47.05, 50.7, 37.7, 31.45, 47.05]
escoxlmr_fijo = [51.3, 44.55, 48.95, 36.5, 31.85, 51.3]
support_fijo = [14, 30, 27, 13, 17]

names = ["SkillSpan", "Sayfullina", "Green", "JobStack", "Gnehm", "Fijo"]


def plot_radar(data, support, names):
    # Set the labels for the chart
    labels = np.array(['{1,2}', '{3,4}', '{5,6}', '{7,8}', '{9,10}'])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, axs = plt.subplots(2, 3, figsize=(10, 8), subplot_kw={'polar': True})
    fig.subplots_adjust(hspace=-0.45)
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        ax.plot(angles, data[0][i], '.-', linewidth=2, color="sandybrown")
        ax.plot(angles, data[1][i], '.-', linewidth=2, color="darkturquoise")
        ax.fill(angles, data[0][i], alpha=0.25, color="sandybrown")
        ax.fill(angles, data[1][i], alpha=0.25, color="darkturquoise")
        ax.set_thetagrids(angles * 180/np.pi, labels, alpha=0.8, fontsize=9)
        for angle, supp in zip(angles, support[i]):
            supp = f"({supp})"
            ax.text(angle, 68, supp, ha='center', va='center', size=9)
        ax.set_ylim(0, 100)
        ax.grid(visible=True, axis="both", which="major", linestyle=":", color="gray")
        ax.set_title(names[i], alpha=0.8, size=10)
    # axs[0].set_ylim(0, 70)
    # axs[2].set_ylim(0, 70)
    # axs[5].set_ylim(0, 70)
    axs[4].legend(labels=["XLM-R", "ESCOXLM-R"], prop={"size": 10}, bbox_to_anchor=(0.5, -.25), loc='lower center',
                  ncol=2)
    fig.tight_layout()
    plt.savefig("annotated.pdf", dpi=300, bbox_inches="tight")


xlmr = [xlmr_skillspan, xlmr_sayfullina, xlmr_green, xlmr_jobstack, xlmr_gnehm, xlmr_fijo]
escoxlmr = [escoxlmr_skillspan, escoxlmr_sayfullina, escoxlmr_green, escoxlmr_jobstack, escoxlmr_gnehm, escoxlmr_fijo]
support = [support_skillspan, support_sayfullina, support_green, support_jobstack, support_gnehm, support_fijo]

plot_radar([xlmr, escoxlmr], support, names)