# -*- coding: utf-8 -*-
"""
Figuras do resumo "Simetria de reticulados e anisotropia de potenciais".
Convenções (iguais às do texto):
  r² = yᵀ G y, com G_Z2 = I e G_A2 = [[2,-1],[-1,2]];
  coordenadas cartesianas x = B y, com BᵀB = G (b1 ao longo do eixo x);
  vetores mínimos do dual: S_Z2 = {(1,0),(0,1)}, S_A2 = {(1,0),(0,1),(1,-1)};
  em cartesianas k = B^{-T} k_lat, de modo que k·x = k_lat·y.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 10.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

# ── reticulados ──────────────────────────────────────────────────────────────
def lattice(G, S):
    G = np.array(G, float)
    L = np.linalg.cholesky(G)                 # G = L Lᵀ  →  B = Lᵀ  (BᵀB = G)
    B = L.T
    K = np.array(S, float) @ np.linalg.inv(B)  # linhas: k_cart = B^{-T} k_lat
    return B, K

LAT = {
    r"$\mathbb{Z}^2$": lattice([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
    r"$A_2$":          lattice([[2, -1], [-1, 2]], [[1, 0], [0, 1], [1, -1]]),
}
for nome, (B, K) in LAT.items():           # verificação: termo quadrático = r²/2
    th = np.linspace(0, 2*np.pi, 7)[:-1]
    u = np.c_[np.cos(th), np.sin(th)]
    assert np.allclose(((u @ K.T) ** 2).sum(1), 1.0), nome

# ── formas funcionais g(s), com U = U0 Σ_k g(k·x); escritas de forma precisa ──
FUN = {
    "cosseno":            lambda s: 2 * np.sin(s / 2) ** 2,       # 1 − cos s
    "cosseno hiperbólico": lambda s: 2 * np.sinh(s / 2) ** 2,      # cosh s − 1
    "gaussiana":          lambda s: -np.expm1(-s ** 2 / 2),       # 1 − e^{−s²/2}
}
def U(K, g, x, y):
    return sum(g(k[0] * x + k[1] * y) for k in K)

# ═════════════════════════════════════════════════════════════════════════════
# Figura 1 — reticulados e vetores mínimos do dual
# ═════════════════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.1))
def seta(ax, p, cor, lw, ls="-", z=3):
    ax.add_patch(FancyArrowPatch((0, 0), tuple(p), arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, color=cor, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))       # a ponta toca o ponto
for ax, (nome, (B, K)), t in zip(axs, LAT.items(), (3, 5)):
    R = 2.25
    n = np.arange(-8, 9)
    P = np.array([B @ [i, j] for i in n for j in n])            # reticulado direto
    Bd = np.linalg.inv(B).T                                     # base do dual (cartesiana)
    Q = np.array([Bd @ [i, j] for i in n for j in n])           # reticulado dual
    inside = lambda A: A[(np.abs(A[:, 0]) <= R) & (np.abs(A[:, 1]) <= R)]
    P, Q = inside(P), inside(Q)
    ax.plot(P[:, 0], P[:, 1], "o", color="0.45", ms=7, zorder=1)
    ax.plot(Q[:, 0], Q[:, 1], "o", mfc="none", mec="C3", mew=1.3, ms=11, zorder=2)
    for kk in K:                                                # ±k com o comprimento real
        seta(ax, kk, "C3", 1.8, "-", 4)
        seta(ax, -kk, "C3", 1.8, "--", 4)
    for jb, b in enumerate(B.T):                                # base do reticulado direto
        seta(ax, b, "k", 2.0, "-", 5)
        dx, dy = (6, -16) if jb == 0 else ((8, -4) if t == 3 else (-30, 4))
        ax.annotate(rf"$\mathbf{{b}}_{jb+1}$", b, xytext=(dx, dy), textcoords="offset points",
                    fontsize=14, zorder=6)
    ax.set_title(f"{nome}:  {2 * len(K)} vetores mínimos do dual  ($t = {t}$)")
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.7")
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", color="0.45", ms=7, label="reticulado $\\Lambda$"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="C3", mew=1.3, ms=11, label="dual $\\Lambda^*$"),
    Line2D([], [], color="k", lw=2.0, label="base $\\{\\mathbf{b}_1,\\mathbf{b}_2\\}$"),
    Line2D([], [], color="C3", lw=1.8, label="$\\pm\\mathbf{k}$, $\\mathbf{k}\\in S$"),
], loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("Figura1_reticulados_dual.png"); fig.savefig("Figura1_reticulados_dual.pdf")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Figura 2 — curvas de nível perto da origem (2 reticulados × 3 funções)
# ═════════════════════════════════════════════════════════════════════════════
R = 2.2
x = np.linspace(-R, R, 701)
X, Y = np.meshgrid(x, x)
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.4), sharex=True, sharey=True)
for i, (nome, (B, K)) in enumerate(LAT.items()):
    for j, (fn, g) in enumerate(FUN.items()):
        ax = axs[i, j]
        Z = U(K, g, X, Y)
        # níveis em r_ref: valores do potencial ao longo do eixo x em raios fixos
        rr = np.array([0.5, 1.0, 1.5, 2.0])
        lev = np.sort(U(K, g, rr, 0 * rr))
        ax.contourf(X, Y, Z, levels=40, cmap="Blues", alpha=0.85)
        ax.contour(X, Y, Z, levels=lev, colors="k", linewidths=0.9)
        for r0 in rr:
            ax.add_patch(plt.Circle((0, 0), r0, fill=False, ls="--", lw=0.6, color="C3"))
        ax.set_aspect("equal")
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        if i == 0:
            ax.set_title(fn)
        if j == 0:
            ax.set_ylabel(f"{nome}\n$y$")
        if i == 1:
            ax.set_xlabel("$x$")
        ax.tick_params(labelsize=11)
fig.text(0.5, -0.02, "Linhas pretas: curvas de nível do potencial; tracejado vermelho:\n"
         "círculos de mesmo raio (referência isotrópica).", ha="center", fontsize=11)
fig.tight_layout()
fig.savefig("Figura2_curvas_de_nivel.png"); fig.savefig("Figura2_curvas_de_nivel.pdf")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Figura 3 — anisotropia ordem a ordem e anisotropia do potencial completo
# ═════════════════════════════════════════════════════════════════════════════
# (a) termo de ordem n: U_n = c_n r^n Σ_k |k|^n cos^n(Θ − θ_k). A razão
#     anisotropia/radial do termo NÃO depende de c_n, logo é a mesma para as três
#     funções: depende só dos vetores mínimos do dual.
th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
u = np.c_[np.cos(th), np.sin(th)]
orders = np.arange(2, 13, 2)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.2, 3.9))
mk = {r"$\mathbb{Z}^2$": ("s", "C0"), r"$A_2$": ("o", "C3")}
FLOOR = 1e-2
for nome, (B, K) in LAT.items():
    m, c = mk[nome]
    rel = []
    for n in orders:
        f = ((u @ K.T) ** n).sum(1)
        rel.append((f.max() - f.min()) / 2 / f.mean())
    rel = np.array(rel)
    zero = rel < 1e-12
    off = -0.25 if nome == r"$\mathbb{Z}^2$" else 0.25
    axa.semilogy(orders[~zero] + off, rel[~zero], m, color=c, ms=8, label=nome)
    axa.semilogy(orders[zero] + off, np.full(zero.sum(), FLOOR), m, color=c,
                 mfc="white", mew=1.6, ms=8, zorder=5)
# Os termos puramente radiais têm anisotropia exatamente 0, que não existe em escala
# log: são mostrados numa faixa "0" separada do eixo por uma quebra.
YLO, YBRK = FLOOR / 1.8, FLOOR * 2.2          # limite inferior e posição da quebra
axa.set_ylim(YLO, 2)
axa.axhspan(YLO, YBRK, color="0.92", lw=0, zorder=0)
axa.set_yticks([FLOOR, 1e-1, 1e0])
axa.set_yticklabels(["0", "$10^{-1}$", "$10^{0}$"])
axa.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
yb = (np.log(YBRK) - np.log(YLO)) / (np.log(2) - np.log(YLO))   # quebra em fração do eixo
d = 0.016
for xx in (0.0, 1.0):
    for dy in (-0.012, 0.012):
        axa.plot([xx - d, xx + d], [yb + dy - d, yb + dy + d], transform=axa.transAxes,
                 color="k", lw=1.0, clip_on=False, zorder=6)
axa.text(12.55, FLOOR, "termos\nradiais", fontsize=10, color="0.35", va="center", ha="right")
axa.set_xticks(orders)
axa.set_xlabel("ordem $n$ do termo da expansão")
axa.set_ylabel("amplitude angular / radial")
axa.set_title("(a) anisotropia de cada ordem")
axa.legend(loc="lower right", bbox_to_anchor=(1.0, 0.24))
axa.text(0.04, 0.63, "idêntica para cosseno,\ncosseno hiperbólico\ne gaussiana",
         transform=axa.transAxes, va="top", fontsize=10.5)

# (b) potencial completo: δ(r) = [max_Θ U − min_Θ U] / (2⟨U⟩_Θ) em r fixo
rs = np.geomspace(0.05, 2.0, 60)
ls = {"cosseno": "-", "cosseno hiperbólico": "--", "gaussiana": ":"}
for nome, (B, K) in LAT.items():
    m, c = mk[nome]
    for fn, g in FUN.items():
        d = []
        for r0 in rs:
            Uv = U(K, g, r0 * u[:, 0], r0 * u[:, 1])
            d.append((Uv.max() - Uv.min()) / 2 / Uv.mean())
        axb.loglog(rs, d, ls[fn], color=c, lw=1.2)
# retas de referência
axb.loglog(rs[:25], 0.06 * rs[:25] ** 2, color="0.4", lw=0.7)
axb.text(0.07, 0.06 * 0.07 ** 2 * 3.2, r"$\propto r^2$", fontsize=12, color="0.3")
axb.loglog(rs[:25], 1.4e-4 * rs[:25] ** 4, color="0.4", lw=0.7)
axb.text(0.07, 1.4e-4 * 0.07 ** 4 * 4, r"$\propto r^4$", fontsize=12, color="0.3")
axb.set_xlabel("$r$")
axb.set_ylabel(r"anisotropia relativa $\delta(r)$")
axb.set_title("(b) potencial completo")
from matplotlib.lines import Line2D
axb.legend(handles=[Line2D([], [], color="C0", label=r"$\mathbb{Z}^2$"),
                    Line2D([], [], color="C3", label=r"$A_2$")] +
                   [Line2D([], [], color="k", ls=v, label=k) for k, v in ls.items()],
           fontsize=9.5, loc="lower right", ncol=1)
fig.tight_layout()
fig.savefig("Figura3_anisotropia_por_ordem.png"); fig.savefig("Figura3_anisotropia_por_ordem.pdf")
plt.close(fig)

# verificação numérica dos coeficientes de 6ª ordem do A2 (texto: r^6/25920 (10 − cos 6Θ))
B, K = LAT[r"$A_2$"]
f6 = ((u @ K.T) ** 6).sum(1) / 720
print("A2, ordem 6 (cosseno): max, min de f6 =", f6.max(), f6.min(),
      " esperado (10±1)/25920 =", 11 / 25920, 9 / 25920)
print("Θ do máximo de f6 (graus):", np.degrees(th[np.argmax(f6)]))
print("ok")
