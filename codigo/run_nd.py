# -*- coding: utf-8 -*-
"""
run_nd.py — dinâmica confinada dos reticulados de 2 a 8 dimensões
(potencial completo × truncado harmônico + quártico).

No Spyder: abra este arquivo, ajuste a CONFIGURAÇÃO e aperte F5.
run_nd.py e lattice_nd.py devem estar na MESMA pasta (de preferência fora do OneDrive).

Cada tarefa (reticulado, modelo, energia) é salva assim que termina. Se você
interromper, aperte F5 de novo: só roda o que falta. Mudar T, h ou o número de
órbitas refaz apenas as tarefas afetadas.
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
QUICK = False            # True: teste rápido de ponta a ponta (~10 min)
DIMS = [2, 3, 4, 5, 6, 7, 8]
LATTICES = None          # None = todos; ou uma lista, ex. ["A3", "D4", "E8"]
MODELS = ["full", "quartic"]          # completo e truncado r² + r⁴
EPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95]   # E/E_sep (completo)
EPS_PLOT = [0.1, 0.5, 0.95]           # energias com seção de Poincaré
T_MAX = 5000.0           # tempo de integração (unidades de 1/ω, ω = 1)
H = 0.02                 # passo do integrador
N_ORBITS = 40            # órbitas por energia
WORKERS = os.cpu_count() or 1         # processos; use 1 se o Spyder travar
ONLY_ANALYSIS = False    # True: só refaz figuras e relatório com o que já existe
OUT = os.path.join(HERE, "resultados_nd")
# ═══════════════════════════════════════════════════════════════════════════

if QUICK:
    T_MAX, N_ORBITS, EPS = 200.0, 8, [0.05, 0.3, 0.8]
    EPS_PLOT = [0.3]
    OUT = os.path.join(HERE, "resultados_nd_teste")


def main():
    argv = ["--models", *MODELS, "--eps", *map(str, EPS), "--eps-plot", *map(str, EPS_PLOT),
            "--t-max", str(T_MAX), "--h", str(H), "--n-orbits", str(N_ORBITS),
            "--workers", str(WORKERS), "--out", OUT, "--dims", *map(str, DIMS)]
    if LATTICES:
        argv += ["--lattices", *LATTICES]
    if ONLY_ANALYSIS:
        argv += ["--only-analysis"]
    lattice_nd.main(argv)


if __name__ == "__main__":      # obrigatório no Windows (multiprocessing usa 'spawn')
    main()
