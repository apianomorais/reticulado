# -*- coding: utf-8 -*-
"""
run_diff.py — regime de difusão (E > E_sep): passeio aleatório determinístico,
comparação com a TST, com o passeio markoviano e com o passeio de Pólya.

No Spyder: F5 (ou %run run_diff.py). Deve ficar na mesma pasta de lattice_nd.py
e lattice_diff.py. Resultados em resultados_difusao\\ ; pode interromper e retomar.
Tempo estimado: ~30 min com 20 processos.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lattice_diff

# ═══════════════════════════════════ CONFIGURAÇÃO ═══════════════════════════
QUICK = False            # True: teste rápido (~5 min)
LATTICES = ["Z2", "A2",                 # 2D: visualização e Pólya recorrente
            "Z3", "A3",
            "Z4", "A4", "D4",
            "Z6", "A6", "E6",
            "Z8", "A8", "E8"]
EPS = [1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0]   # E/E_sep (> 1)
T_MAX = 20000.0
H = 0.01
N_ORBITS = 60
WORKERS = os.cpu_count() or 1
ONLY_ANALYSIS = False
OUT = os.path.join(HERE, "resultados_difusao")
# ═══════════════════════════════════════════════════════════════════════════

if QUICK:
    LATTICES, EPS, T_MAX, N_ORBITS = ["Z2", "A2", "A3", "D4"], [1.05, 1.5, 3.0], 1000.0, 16
    OUT = os.path.join(HERE, "resultados_difusao_teste")


def main():
    argv = ["--lattices", *LATTICES, "--eps", *map(str, EPS), "--t-max", str(T_MAX),
            "--h", str(H), "--n-orbits", str(N_ORBITS), "--workers", str(WORKERS), "--out", OUT]
    if ONLY_ANALYSIS:
        argv.append("--only-analysis")
    lattice_diff.main(argv)


if __name__ == "__main__":
    main()
