"""
Worker de KGeoMIP. Recibe parámetros JSON por argv[1], imprime resultado JSON a stdout.
Ejecutado como subprocess aislado por run_batch.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_arg = sys.argv[1]
if _arg.startswith("@"):
    with open(_arg[1:], encoding="utf-8") as _f:
        params = json.loads(_f.read())
else:
    params = json.loads(_arg)
N              = params["N"]
estado_inicial = params["estado_inicial"]
condicion_bin  = params["condicion_bin"]
alcance_bin    = params["alcance_bin"]
mecanismo_bin  = params["mecanismo_bin"]
ks             = params["ks"]
semilla        = params["semilla"]

GEOMIP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "GeoMIP" / "src" / "Method2_Dynamic_Programming_Reformulation"
)
sys.path.insert(0, str(GEOMIP_ROOT))

if "tpm_data" in params:
    tpm = np.array(params["tpm_data"]).reshape(2**N, N).astype(float)
else:
    np.random.seed(semilla)
    tpm = np.random.randint(2, size=(2**N, N), dtype=np.int8).astype(float)

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import KGeoMIP

gestor = Manager(estado_inicial=estado_inicial)
resultados = KGeoMIP(gestor).aplicar_estrategia(
    condicion=condicion_bin,
    alcance=alcance_bin,
    mecanismo=mecanismo_bin,
    tpm=tpm,
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
