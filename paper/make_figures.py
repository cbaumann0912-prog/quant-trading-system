import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK = "#14181D"
ACCENT = "#1F3A4D"
MUTE = "#8B939D"
GRID = "#DDE0E5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": MUTE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
})

def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(5.4, 3.2))
yrs = np.linspace(10, 25, 400)
ax.plot(yrs, 2.0 / np.sqrt(yrs), color=ACCENT, linewidth=2, zorder=3)

pts = [(13.0, 0.555), (18.4, 0.466), (21.8, 0.428), (23.6, 0.412)]
for x, y in pts:
    ax.plot([x], [y], "o", ms=5.5, color=ACCENT, mec="white", mew=1.2, zorder=4)

ax.annotate("As downloaded\n10 pairs from 2011\n$SR = 0.555$",
            xy=(13.0, 0.555), xytext=(14.4, 0.632),
            fontsize=7.4, color=INK, linespacing=1.4, va="top",
            arrowprops=dict(arrowstyle="-", color=MUTE, linewidth=0.8,
                            shrinkA=0, shrinkB=4))
ax.annotate("Drop NZD/USD\n9 pairs from 2002\n$SR = 0.428$",
            xy=(21.8, 0.428), xytext=(15.3, 0.4225),
            fontsize=7.4, color=INK, linespacing=1.4, va="top",
            arrowprops=dict(arrowstyle="-", color=MUTE, linewidth=0.8,
                            shrinkA=0, shrinkB=4))
ax.annotate("", xy=(13.0, 0.555), xytext=(13.0, 0.428),
            arrowprops=dict(arrowstyle="<->", color=MUTE, linewidth=0.9))
ax.text(13.35, 0.4905, "23%", fontsize=7.4, color=MUTE, va="center")
ax.hlines(0.428, 13.0, 21.8, color=MUTE, linewidth=0.7,
          linestyle=(0, (3, 3)), zorder=1)

ax.set_xlabel("Years of development data")
ax.set_ylabel("True Sharpe required to reach $t=2$")
ax.set_xlim(10, 25.6)
ax.set_ylim(0.368, 0.665)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
frame(ax)
fig.tight_layout()
fig.savefig("figures/fig1_power.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


fig, ax = plt.subplots(figsize=(5.4, 3.2))
rho = np.linspace(0.0, 1.0, 400)
N = 10
ax.plot(rho, N / (1 + (N - 1) * rho), color=ACCENT, linewidth=2, zorder=3)

r0, b0 = 0.3652, 2.33
ax.plot([r0], [b0], "o", ms=6, color=ACCENT, mec="white", mew=1.2, zorder=4)
ax.vlines(r0, 0, b0, color=MUTE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
ax.hlines(b0, 0, r0, color=MUTE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
ax.text(r0 + 0.025, b0 + 0.45,
        "Observed book\n$\\rho = 0.3652$, breadth $2.33$",
        fontsize=7.6, color=INK, va="bottom", linespacing=1.4)

ax.set_xlabel("Mean pairwise correlation of book components")
ax.set_ylabel("Effective breadth,  $N/(1+(N-1)\\rho)$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 10.4)
ax.set_yticks([0, 2, 4, 6, 8, 10])
frame(ax)
fig.tight_layout()
fig.savefig("figures/fig2_breadth.png", bbox_inches="tight", facecolor="white")
plt.close(fig)


fig, ax = plt.subplots(figsize=(5.4, 3.05))
pairs = ["USD/JPY", "EUR/USD", "GBP/USD"]
agree = [20.52, 49.41, 55.74]
flips = ["1,739 of 2,188", "1,107 of 2,188", "968 of 2,187"]
y = np.arange(len(pairs))

ax.barh(y, agree, height=0.5, color=ACCENT, zorder=3)
for yi, a, f in zip(y, agree, flips):
    ax.text(a + 1.4, yi, f"{a:.2f}%   ({f} bars flip)", va="center",
            fontsize=7.6, color=INK)

ax.axvline(33.3, color=MUTE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
ax.text(34.6, 2.95, "chance agreement among three labels",
        fontsize=7.2, color=MUTE, ha="left", va="center")

ax.set_yticks(y, pairs)
ax.set_xlabel("Out-of-sample days on which the two classifiers agree (%)")
ax.set_xlim(0, 82)
ax.set_ylim(-0.6, 3.35)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig("figures/fig3_leakage.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("figures written")
