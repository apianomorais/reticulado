# -*- coding: utf-8 -*-
"""GIF: mesma condição inicial nos reticulados ℤ² e A₂ (potencial cossenoidal).
No ℤ² o termo r⁴ anisotrópico exerce torque → L oscila e troca de sinal.
No A₂ o termo r⁴ é radial → L se conserva (só o r⁶ quebra, muito fracamente)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 12,
                     "axes.titlesize": 14, "axes.labelsize": 13})

def dual(G, S):
    B = np.linalg.cholesky(np.array(G, float)).T
    return np.array(S, float) @ np.linalg.inv(B)
LAT = {r"$\mathbb{Z}^2$": dual([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
       r"$A_2$": dual([[2, -1], [-1, 2]], [[1, 0], [0, 1], [1, -1]])}
COR = {r"$\mathbb{Z}^2$": "C0", r"$A_2$": "C3"}

E, T, h, ANG = 0.5, 420.0, 0.005, 0.3
U = lambda K, x, y: sum(1 - np.cos(k[0] * x + k[1] * y) for k in K)

def integra(K):
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
t = np.arange(len(next(iter(DATA.values()))[1])) * h * 10
L0 = DATA[r"$\mathbb{Z}^2$"][1][0]

fig = plt.figure(figsize=(9, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.15)
axo = {nm: fig.add_subplot(gs[0, j]) for j, nm in enumerate(LAT)}
axL = fig.add_subplot(gs[1, :])
xg = np.linspace(-1.35, 1.35, 300); X, Y = np.meshgrid(xg, xg)
art = {}
for nm, ax in axo.items():
    ax.contour(X, Y, U(LAT[nm], X, Y), levels=[E], colors="0.5", linestyles="--", linewidths=1)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(nm + (":  termo $r^4$ anisotrópico" if "Z" in nm else ":  termo $r^4$ radial"))
    fundo, = ax.plot([], [], color=COR[nm], lw=0.4, alpha=0.25)
    rastro, = ax.plot([], [], color=COR[nm], lw=1.8)
    ponto, = ax.plot([], [], "o", color="k", ms=6)
    art[nm] = (fundo, rastro, ponto)
for nm in LAT:
    axL.plot(t, DATA[nm][1] / L0, color=COR[nm], lw=0.6, alpha=0.2)
curvas = {nm: axL.plot([], [], color=COR[nm], lw=2, label=nm)[0] for nm in LAT}
axL.axhline(0, color="0.6", lw=0.6)
axL.set_xlim(0, T); axL.set_ylim(-1.6, 1.6)
axL.set_xlabel("tempo $t$"); axL.set_ylabel(r"$L(t)\,/\,L(0)$")
axL.set_title("momento angular  $L = xP_y - yP_x$")
axL.legend(loc="lower left", ncol=2)
cursor = axL.axvline(0, color="k", lw=0.8)
fig.suptitle(f"Mesma condição inicial, potencial cossenoidal, $E = {E}\\,\\Phi_0$", y=0.985)
fig.text(0.5, 0.005, "tracejado: curva $U = E$ (limite do movimento)", ha="center", fontsize=10,
         color="0.35")

NF, TRAIL = 240, int(12 / (h * 10))
idx = np.linspace(1, len(t) - 1, NF).astype(int)
def quadro(f):
    i = idx[f]
    for nm, (fundo, rastro, ponto) in art.items():
        R, L = DATA[nm]
        fundo.set_data(R[:i, 0], R[:i, 1])
        j0 = max(0, i - TRAIL)
        rastro.set_data(R[j0:i, 0], R[j0:i, 1])
        ponto.set_data([R[i, 0]], [R[i, 1]])
        curvas[nm].set_data(t[:i], L[:i] / L0)
    cursor.set_xdata([t[i], t[i]])
    return []
anim = FuncAnimation(fig, quadro, frames=NF, blit=False)
anim.save("momento_angular_Z2_A2.gif", writer=PillowWriter(fps=20), dpi=90)
quadro(NF - 1); fig.savefig("momento_angular_Z2_A2_final.png", dpi=300, bbox_inches="tight")
print("ok")
