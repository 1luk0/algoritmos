"""
Worker de KQNodes. Recibe parámetros JSON por argv[1], imprime resultado JSON a stdout.
Ejecutado como subprocess aislado por run_batch.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_arg = sys.argv[1]
if _arg.startswith("@"):
    with open(_arg[1:], encoding="utf-8") as _f:
        params = json.loads(_f.read())
else:
    params = json.loads(_arg)
N             = params["N"]
estado_inicial = params["estado_inicial"]
condicion_bin  = params["condicion_bin"]
alcance_bin    = params["alcance_bin"]
mecanismo_bin  = params["mecanismo_bin"]
ks             = params["ks"]
semilla        = params["semilla"]

QNODES_ROOT = Path(__file__).resolve().parents[1] / "QNodes"
sys.path.insert(0, str(QNODES_ROOT))

if "tpm_data" in params:
    tpm = np.array(params["tpm_data"]).reshape(2**N, N).astype(float)
else:
    np.random.seed(semilla)
    tpm = np.random.randint(2, size=(2**N, N), dtype=np.int8).astype(float)

from src.strategies.q_nodes import KQNodes

q = KQNodes(tpm)
t0 = time.time()
resultados = q.aplicar_estrategia(
    estado_inicial,
    condicion_bin,
    alcance_bin,
    mecanismo_bin,
    ks=ks,
)

salida = {}
for k, sol in resultados.items():
    salida[str(k)] = {
        "particion": sol.particion,
        "perdida":   float(sol.perdida),
        "tiempo":    float(sol.tiempo_ejecucion),
    }

print(json.dumps(salida, ensure_ascii=False))
