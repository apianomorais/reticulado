# -*- coding: utf-8 -*-
"""
figura_sintese_nd.py — figura-síntese a partir de resultados_nd.csv.

(a) δ|L|² × E/E_sep, potencial completo: classes separadas; para t ≥ 5 a quebra da
    isotropia segue δ|L|² ∝ E^((t−3)/2) (retas tracejadas: ajuste com expoente fixo).
(b) δ|L|² × E/E_sep, potencial quártico: t ≥ 5 (e A₂) conservam |L|² exatamente.
(c) λ₁ / piso(ℤᴺ) × N em E/E_sep fixo, por classe.

Uso (Spyder): F5 com CSV = caminho do resultados_nd.csv; ou
    python figura_sintese_nd.py caminho/resultados_nd.csv
"""
import csv
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "resultados_nd", "resultados_nd.csv")
E_FIX = (0.02, 0.05)          # energias do painel (c)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 12,
                     "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 10.5})
COR = {"separável": "0.55", "N = 2": "C2", "t = 3": "C0", "t = 5": "C3", "t = 7": "C4"}


def carregar(path):
    D = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            D.setdefault((r["reticulado"], r["modelo"]), []).append(r)
    out = {}
    for k, rows in D.items():
        rows.sort(key=lambda r: float(r["E/E_sep"]))
        out[k] = dict(N=int(rows[0]["N"]), classe=rows[0]["classe"], t=int(rows[0]["t"]),
                      **{f: np.array([float(r[f]) for r in rows])
                         for f in ("E/E_sep", "lam_med", "dL2_med")})
    return out


def main(path=CSV):
    R = carregar(path)
    nomes = sorted({k for k, _ in R}, key=lambda s: (int(s[1]), s))
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # (a) e (b): δ|L|²
    for ax, mod, tit in ((axs[0], "completo", "(a) potencial completo"),
                         (axs[1], "quártico", r"(b) potencial quártico ($r^2 + r^4$)")):
        for n in nomes:
            z = R.get((n, mod))
            if z is None:
                continue
            ax.plot(z["E/E_sep"], z["dL2_med"], "o-", ms=3, lw=1.1, alpha=0.85,
                    color=COR.get(z["classe"], "k"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$E/E_{sep}$")
        ax.set_title(tit)
    axs[0].set_ylabel(r"variação de $|L|^2$:  $\delta|L|^2$ (mediana)")
    # retas E^((t−3)/2) para t = 5 e 7, ajustadas só no prefator (E/E_sep ≤ 0,3)
    for t, cor in ((5, COR["t = 5"]), (7, COR["t = 7"])):
        q = (t - 3) / 2
        pref = []
        for n in nomes:
            z = R.get((n, "completo"))
            if z is not None and z["t"] == t and z["classe"] != "N = 2":
                m = z["E/E_sep"] <= 0.3
                pref.append(np.mean(np.log(z["dL2_med"][m]) - q * np.log(z["E/E_sep"][m])))
        if pref:
            e = np.geomspace(0.008, 0.4, 50)
            axs[0].plot(e, np.exp(np.mean(pref)) * e ** q / 3, "--", color=cor, lw=1.6)
            axs[0].annotate(rf"$\propto E^{{{q:g}}}$", (0.012, np.exp(np.mean(pref)) * 0.012 ** q / 3),
                            xytext=(0, -16), textcoords="offset points", color=cor, fontsize=12)
    axs[1].axhspan(1e-14, 1e-11, color="0.92", zorder=0)
    axs[1].text(0.012, 3e-12, "conservação exata (precisão de máquina)", fontsize=10,
                color="0.3")

    # (c) λ / piso × N
    ax = axs[2]
    mk = {E_FIX[0]: "s", E_FIX[1]: "o"}
    for ef in E_FIX:
        for n in nomes:
            z = R.get((n, "completo"))
            if z is None or z["classe"] == "separável":
                continue
            zz = R.get((f"Z{z['N']}", "completo"))
            i, j = np.isclose(z["E/E_sep"], ef), np.isclose(zz["E/E_sep"], ef)
            if not (i.any() and j.any()):
                continue
            v = z["lam_med"][i][0] / zz["lam_med"][j][0]
            dx = {"t = 3": -0.12, "N = 2": 0.0, "t = 5": 0.12, "t = 7": 0.12}[z["classe"]]
            dx += -0.05 if ef == E_FIX[0] else 0.05
            ax.plot(z["N"] + dx, v, mk[ef], color=COR[z["classe"]], ms=7,
                    mfc=COR[z["classe"]] if ef == E_FIX[1] else "white", mew=1.5)
    ax.axhline(1, color="0.5", ls=":", lw=1)
    ax.text(1.75, 1.08, r"piso ($\mathbb{Z}^N$)", fontsize=10, color="0.4", ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("dimensão $N$")
    ax.set_xticks(range(2, 9))
    ax.set_xlim(1.6, 8.5)
    ax.set_ylabel(r"$\lambda_1\,/\,\lambda_1(\mathbb{Z}^N)$")
    ax.set_title("(c) caos a baixa energia")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label=f"E/E_sep = {E_FIX[1]:g}"),
                       Line2D([], [], marker="s", ls="", color="k", mfc="white",
                              label=f"E/E_sep = {E_FIX[0]:g}")], loc="upper right")

    fig.legend(handles=[Line2D([], [], color=c, lw=2.5, label=k) for k, c in COR.items()],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "figura_sintese_nd")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figura salva em", out + ".png")
    return out + ".png"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV)
