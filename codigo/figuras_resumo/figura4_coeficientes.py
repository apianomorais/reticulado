# -*- coding: utf-8 -*-
"""Figura 4 — coeficientes da expansão U = U0 [c_rad r^n + c_ang r^n cos(mΘ)] por ordem,
reticulado e forma funcional (Θ medido a partir de b1). Mesmas convenções das Figuras 1–3."""
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 10.5,
    "xtick.labelsize": 12, "ytick.labelsize": 11, "savefig.dpi": 400, "savefig.bbox": "tight",
})

def dual_min(G, S):
    B = np.linalg.cholesky(np.array(G, float)).T
    return np.array(S, float) @ np.linalg.inv(B)
LAT = {r"$\mathbb{Z}^2$": (dual_min([[1, 0], [0, 1]], [[1, 0], [0, 1]]), 4),
       r"$A_2$": (dual_min([[2, -1], [-1, 2]], [[1, 0], [0, 1], [1, -1]]), 6)}
# g(s) = Σ a_n s^n  (n = 2, 4, 6)
FUN = {"cosseno": {2: Fraction(1, 2), 4: Fraction(-1, 24), 6: Fraction(1, 720)},
       "cosseno hiperbólico": {2: Fraction(1, 2), 4: Fraction(1, 24), 6: Fraction(1, 720)},
       "gaussiana": {2: Fraction(1, 2), 4: Fraction(-1, 8), 6: Fraction(1, 48)}}
COR = {"cosseno": "C0", "cosseno hiperbólico": "C2", "gaussiana": "C1"}

th = np.linspace(0, 2 * np.pi, 7200, endpoint=False)
u = np.c_[np.cos(th), np.sin(th)]

def angular_sum(K, n):
    """Σ_k (k·r̂)^n = A + C cos(mΘ): devolve (A, C) exatos (frações)."""
    f = ((u @ K.T) ** n).sum(1)
    A = Fraction(f.mean()).limit_denominator(10000)
    C = Fraction(f[0] - f.mean()).limit_denominator(10000)
    return A, C

groups = [(2, "rad"), (4, "rad"), (4, "ang"), (6, "rad"), (6, "ang")]
glabel = {(2, "rad"): "$r^2$\nradial", (4, "rad"): "$r^4$\nradial", (4, "ang"): "$r^4$\nangular",
          (6, "rad"): "$r^6$\nradial", (6, "ang"): "$r^6$\nangular"}

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
w = 0.26
for ax, (lat, (K, m)) in zip(axs, LAT.items()):
    for gi, (n, kind) in enumerate(groups):
        A, C = angular_sum(K, n)
        base = A if kind == "rad" else C
        for fi, (fn, coef) in enumerate(FUN.items()):
            c = coef[n] * base
            x = gi + (fi - 1) * w
            print(f"{lat:16s} n={n} {kind}  {fn:20s} {c}")
            if c == 0:
                ax.plot(x, 2e-5, "x", color=COR[fn], ms=9, mew=2)
                continue
            neg = c < 0
            ax.bar(x, abs(float(c)), w * 0.92, color="white" if neg else COR[fn],
                   edgecolor=COR[fn], hatch="////" if neg else None, lw=1.4)
            s = f"{'−' if neg else '+'}{abs(c.numerator)}/{c.denominator}" if c.denominator != 1 \
                else f"{c}"
            ax.text(x, abs(float(c)) * 1.35, s, rotation=90, ha="center", va="bottom", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(1.2e-5, 30)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([glabel[g] for g in groups])
    ax.set_title(f"{lat}:  parte angular  $\\propto \\cos({m}\\Theta)$")
    for gi, (n, kind) in enumerate(groups):
        if kind == "ang":
            ax.axvspan(gi - 0.45, gi + 0.45, color="0.93", zorder=0)
axs[0].set_ylabel("|coeficiente|  (unidades de $U_0$)")
fig.legend(handles=[Patch(facecolor=COR[f], edgecolor=COR[f], label=f) for f in FUN] +
           [Patch(facecolor="white", edgecolor="0.3", hatch="////", label="coeficiente negativo"),
            Line2D([], [], marker="x", ls="", color="0.3", ms=9, mew=2, label="coeficiente nulo")],
           loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("Figura4_coeficientes.png")
fig.savefig("Figura4_coeficientes.pdf")
plt.close(fig)

