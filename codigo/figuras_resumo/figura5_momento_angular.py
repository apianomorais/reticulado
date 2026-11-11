# -*- coding: utf-8 -*-
"""Figura 5 (estática) e GIF: mesma condição inicial nos reticulados ℤ² e A₂,
potencial cossenoidal. No ℤ² o termo r⁴ anisotrópico exerce torque → L oscila e troca
de sinal. No A₂ o termo r⁴ é radial → L se conserva (só o r⁶ o quebra, fracamente)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 12,
                     "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

def dual(G, S):
    B = np.linalg.cholesky(np.array(G, float)).T
    return np.array(S, float) @ np.linalg.inv(B)
LAT = {r"$\mathbb{Z}^2$": dual([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
       r"$A_2$": dual([[2, -1], [-1, 2]], [[1, 0], [0, 1], [1, -1]])}
COR = {r"$\mathbb{Z}^2$": "C0", r"$A_2$": "C3"}
E, T, h, ANG = 0.5, 420.0, 0.005, 0.3
U = lambda K, x, y: sum(1 - np.cos(k[0] * x + k[1] * y) for k in K)

def integra(K):                                   # Verlet; L = x P_y − y P_x
    r = np.array([0.3 * np.sqrt(2 * E), 0.0])
    v = np.sqrt(2 * (E - U(K, *r))) * np.array([np.sin(ANG), np.cos(ANG)])
    g = lambda r: np.sin(K @ r) @ K
    n, s = int(T / h), 10
    R = np.empty((n // s, 2)); L = np.empty(n // s)
    a = g(r)
    for i in range(n):
        if i % s == 0:
            R[i // s] = r; L[i // s] = r[0] * v[1] - r[1] * v[0]
        v -= 0.5 * h * a; r += h * v; a = g(r); v -= 0.5 * h * a
    return R, L
DATA = {nm: integra(K) for nm, K in LAT.items()}
t = np.arange(len(DATA[r"$A_2$"][1])) * h * 10
L0 = DATA[r"$\mathbb{Z}^2$"][1][0]
for nm, (R, L) in DATA.items():
    print(f"{nm}: L/L0 ∈ [{L.min()/L0:.3f}, {L.max()/L0:.3f}]")

xg = np.linspace(-1.35, 1.35, 300); X, Y = np.meshgrid(xg, xg)
def montar(figsize):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.28, wspace=0.12)
    axo = {nm: fig.add_subplot(gs[0, j]) for j, nm in enumerate(LAT)}
    axL = fig.add_subplot(gs[1, :])
    for k, (nm, ax) in enumerate(axo.items()):
        ax.contour(X, Y, U(LAT[nm], X, Y), levels=[E], colors="0.45", linestyles="--",
                   linewidths=1)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'ab'[k]}) " + nm + (": termo $r^4$ anisotrópico" if k == 0
                                              else ": termo $r^4$ radial"))
    axL.axhline(0, color="0.6", lw=0.6)
    axL.set_xlim(0, T); axL.set_ylim(-1.65, 2.15)
    axL.set_xlabel("tempo $t$  (unidades de $\\omega_0^{-1}$)")
    axL.set_ylabel(r"$L(t)\,/\,L(0)$")
    axL.set_title("(c) momento angular  $L = xP_y - yP_x$")
    return fig, axo, axL

# ── Figura 5 estática ────────────────────────────────────────────────────────
fig, axo, axL = montar((7.6, 6.6))
for nm, ax in axo.items():
    R, L = DATA[nm]
    ax.plot(R[:, 0], R[:, 1], color=COR[nm], lw=0.35, alpha=0.75)
    ax.plot(*R[0], "ko", ms=5)
    axL.plot(t, L / L0, color=COR[nm], lw=1.6, label=nm)
axL.legend(loc="upper right", ncol=2, framealpha=1)
fig.savefig("Figura5_momento_angular.png", dpi=400, bbox_inches="tight")
fig.savefig("Figura5_momento_angular.pdf", bbox_inches="tight")
plt.close(fig)

# ── GIF ──────────────────────────────────────────────────────────────────────
fig, axo, axL = montar((8.4, 7.3))
art = {}
for nm, ax in axo.items():
    fundo, = ax.plot([], [], color=COR[nm], lw=0.4, alpha=0.3)
    rastro, = ax.plot([], [], color=COR[nm], lw=2)
    ponto, = ax.plot([], [], "o", color="k", ms=6)
    art[nm] = (fundo, rastro, ponto)
    axL.plot(t, DATA[nm][1] / L0, color=COR[nm], lw=0.6, alpha=0.18)
curvas = {nm: axL.plot([], [], color=COR[nm], lw=2, label=nm)[0] for nm in LAT}
axL.legend(loc="upper right", ncol=2, framealpha=1)
cursor = axL.axvline(0, color="k", lw=0.8)
fig.suptitle(f"Mesma condição inicial, potencial cossenoidal, $E = {E}\\,\\Phi_0$", y=0.98)
NF, TRAIL = 160, int(12 / (h * 10))
idx = np.linspace(1, len(t) - 1, NF).astype(int)
def quadro(f):
    i = idx[f]
    for nm, (fundo, rastro, ponto) in art.items():
        R, L = DATA[nm]
        fundo.set_data(R[:i, 0], R[:i, 1])
        rastro.set_data(R[max(0, i - TRAIL):i, 0], R[max(0, i - TRAIL):i, 1])
        ponto.set_data([R[i, 0]], [R[i, 1]])
        curvas[nm].set_data(t[:i], L[:i] / L0)
    cursor.set_xdata([t[i], t[i]])
    return []
FuncAnimation(fig, quadro, frames=NF).save("momento_angular_Z2_A2.gif",
                                           writer=PillowWriter(fps=16), dpi=62)
plt.close(fig)
print("ok")
