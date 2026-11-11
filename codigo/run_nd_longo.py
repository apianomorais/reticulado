# -*- coding: utf-8 -*-
"""
run_nd_longo.py — rodada longa a BAIXA ENERGIA (potencial completo).

Objetivo: com T = 5·10⁴, λT passa de ~5 para ~50 nos casos t = 3, o que torna a
fração caótica (SALI) e o λ em janelas confiáveis abaixo de 0,1·E_sep.
Reticulados: um t = 3 e um t ≥ 5 por dimensão, mais o ℤⁿ correspondente,
que fornece o piso de ruído do λ (sistema separável, integrável) no mesmo T.

No Spyder: abra este arquivo e aperte F5 (ou, no console: %run run_nd_longo.py).
Deve ficar na mesma pasta de lattice_nd.py. Os resultados vão para
resultados_nd_longo\ (separados da rodada principal). Pode interromper e retomar.
Tempo estimado: ~13 núcleo·h, cerca de 1 h com 20 processos (as tarefas do E₈
são as mais longas, ~50 min cada).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lattice_nd

# ═══════════════════════════════════ CONFIGURAÇÃO ═══════════════════════════
LATTICES = ["Z3", "A3",            # N = 3: t = 3 (sem t ≥ 5 em N = 3)
            "Z4", "A4", "D4",      # N = 4: t = 3 e t = 5
            "Z6", "A6", "E6",      # N = 6
            "Z8", "A8", "E8"]      # N = 8: t = 3 e t = 7
EPS = [0.005, 0.01, 0.02, 0.05, 0.1]
T_MAX = 50000.0
H = 0.02
N_ORBITS = 40
WORKERS = os.cpu_count() or 1
OUT = os.path.join(HERE, "resultados_nd_longo")
# ═══════════════════════════════════════════════════════════════════════════


def main():
    lattice_nd.main(["--lattices", *LATTICES, "--models", "full",
                     "--eps", *map(str, EPS), "--eps-plot", "0.05",
                     "--t-max", str(T_MAX), "--h", str(H), "--n-orbits", str(N_ORBITS),
                     "--workers", str(WORKERS), "--out", OUT])


if __name__ == "__main__":
    main()
