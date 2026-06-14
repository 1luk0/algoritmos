"""
Orquestador de pruebas por lotes.

Uso:
  python tests/run_batch.py --sheet 10A --desde 1 --hasta 25 --strategy qnodes
  python tests/run_batch.py --sheet 25A --strategy geomip --safe-only
  python tests/run_batch.py --sheet 15B --strategy qnodes  --ks 2 3 4 5 --timeout 300

Argumentos:
  --sheet      Hoja del Excel: 10A | 15B | 20A | 22A | 25A
  --strategy   qnodes | geomip
  --desde      Número de prueba inicial (default 1)
  --hasta      Número de prueba final inclusivo (default: todas)
  --ks         k-particiones a calcular (default: 2 3 4 5)
  --timeout    Segundos máximos por prueba (default 300)
  --safe-only  Solo pruebas con mecanismo ≤ 17 nodos (relevante para 25A con 16 GB RAM)

Salida: results/{strategy}_{sheet}_d{desde}_h{hasta}.xlsx
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parents[1]
EXCEL     = ROOT / "docs" / "DatosPruebas2026_1.xlsx"
RESULTS   = ROOT / "results"
SEMILLA   = 73

ABECEDARY = "ABCDEFGHIJKLMNOPQRSTUVWXY"

SHEET_CONFIG = {
    "10A": {"N": 10, "sheet": "10A-Elementos",   "estado_inicial": "1000000000"},
    "15B": {"N": 15, "sheet": "15B-Elementos",   "estado_inicial": "100000000000000"},
    "20A": {"N": 20, "sheet": "20A-Elementos",   "estado_inicial": "10000000000000000000"},
    "22A": {"N": 22, "sheet": "22A-Elementos",   "estado_inicial": "1000000000000000000000"},
    "25A": {"N": 25, "sheet": "25A-Elementos ",  "estado_inicial": "1000000000000000000000000"},
}

WORKER = {
    "qnodes":            ROOT / "gui" / "logica" / "_worker_qnodes.py",
    "geomip":            ROOT / "gui" / "logica" / "_worker_geomip.py",
    "geomip-recursivo":  ROOT / "gui" / "logica" / "_worker_geomip_recursivo.py",
}

# Umbral de mecanismo seguro para 25A con 16 GB RAM
SAFE_MECH_THRESHOLD = 17


# ─── Utilidades ──────────────────────────────────────────────────────────────

def letras_a_binario(texto: str, n_bits: int) -> str:
    b = ["0"] * n_bits
    for c in texto:
        p = ABECEDARY.find(c)
        if 0 <= p < n_bits:
            b[p] = "1"
    return "".join(b)


def leer_pruebas(sheet_key: str) -> list[dict]:
    cfg = SHEET_CONFIG[sheet_key]
    N   = cfg["N"]
    df  = pd.read_excel(EXCEL, sheet_name=cfg["sheet"], skiprows=4, header=0)
    df  = df.dropna(subset=[df.columns[0]])
    df.columns = ["prueba", "alcance", "mecanismo"] + list(df.columns[3:])
    pruebas = []
    for _, row in df.iterrows():
        alc = str(row["alcance"]).strip()
        mec = str(row["mecanismo"]).strip()
        pruebas.append({
            "prueba":        int(row["prueba"]),
            "alcance":       alc,
            "mecanismo":     mec,
            "alcance_bin":   letras_a_binario(alc, N),
            "mecanismo_bin": letras_a_binario(mec, N),
            "condicion_bin": "1" * N,
            "n_mecanismo":   len(mec),
        })
    return pruebas


def es_segura(prueba: dict, sheet_key: str) -> bool:
    if sheet_key != "25A":
        return True
    return prueba["n_mecanismo"] <= SAFE_MECH_THRESHOLD


# ─── Ejecución de una prueba ──────────────────────────────────────────────────

def ejecutar_prueba(
    prueba: dict,
    sheet_key: str,
    strategy: str,
    ks: list[int],
    timeout: int,
) -> dict:
    cfg = SHEET_CONFIG[sheet_key]
    params = json.dumps({
        "N":             cfg["N"],
        "estado_inicial": cfg["estado_inicial"],
        "condicion_bin": prueba["condicion_bin"],
        "alcance_bin":   prueba["alcance_bin"],
        "mecanismo_bin": prueba["mecanismo_bin"],
        "ks":            ks,
        "semilla":       SEMILLA,
    }, ensure_ascii=False)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER[strategy]), params],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            stderr_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "error desconocido"
            status = "oom" if "MemoryError" in proc.stderr else "error"
            return {"status": status, "error": stderr_msg, "tiempo_total": elapsed}

        resultado = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"status": "ok", "resultado": resultado, "tiempo_total": elapsed}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tiempo_total": timeout}
    except Exception as e:
        return {"status": "error", "error": str(e), "tiempo_total": time.time() - t0}


# ─── Construcción de filas de resultado ──────────────────────────────────────

def filas_de_resultado(prueba: dict, sheet_key: str, strategy: str, ejecucion: dict) -> list[dict]:
    base = {
        "sheet":    sheet_key,
        "strategy": strategy,
        "prueba":   prueba["prueba"],
        "alcance":  prueba["alcance"],
        "mecanismo": prueba["mecanismo"],
        "n_mec":    prueba["n_mecanismo"],
    }
    filas = []
    if ejecucion["status"] == "ok":
        for k_str, datos in ejecucion["resultado"].items():
            filas.append({**base,
                "k":        int(k_str),
                "particion": datos["particion"],
                "perdida":   datos["perdida"],
                "tiempo":    datos["tiempo"],
                "status":    "ok",
                "error":     "",
            })
    else:
        for k in [2, 3, 4, 5]:
            filas.append({**base,
                "k":        k,
                "particion": "",
                "perdida":   None,
                "tiempo":    ejecucion.get("tiempo_total"),
                "status":    ejecucion["status"],
                "error":     ejecucion.get("error", ""),
            })
    return filas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ejecutor de pruebas IIT por lotes")
    parser.add_argument("--sheet",     required=True, choices=list(SHEET_CONFIG))
    parser.add_argument("--strategy",  required=True, choices=["qnodes", "geomip", "geomip-recursivo"])
    parser.add_argument("--desde",     type=int, default=1)
    parser.add_argument("--hasta",     type=int, default=None)
    parser.add_argument("--pruebas",   type=int, nargs="+", default=None,
                        help="Lista explícita de números de prueba a ejecutar")
    parser.add_argument("--ks",        type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--timeout",   type=int, default=300,
                        help="Segundos máximos por prueba (0 = sin límite)")
    parser.add_argument("--safe-only", action="store_true",
                        help="Solo pruebas con mecanismo <= 17 nodos (25A con 16 GB RAM)")
    parser.add_argument("--tag",      type=str, default=None,
                        help="Sufijo adicional para el nombre del archivo de salida (ej. opt)")
    args = parser.parse_args()

    pruebas = leer_pruebas(args.sheet)

    # Filtrar por lista explícita o por rango numérico
    if args.pruebas:
        pruebas = [p for p in pruebas if p["prueba"] in args.pruebas]
    else:
        pruebas = [p for p in pruebas if p["prueba"] >= args.desde]
        if args.hasta:
            pruebas = [p for p in pruebas if p["prueba"] <= args.hasta]

    # Filtrar seguras si se indica
    if args.safe_only:
        antes   = len(pruebas)
        pruebas = [p for p in pruebas if es_segura(p, args.sheet)]
        print(f"[safe-only] {len(pruebas)}/{antes} pruebas seleccionadas "
              f"(mecanismo <= {SAFE_MECH_THRESHOLD} nodos)")

    if not pruebas:
        print("No hay pruebas que ejecutar con los filtros indicados.")
        return

    desde_real = pruebas[0]["prueba"]
    hasta_real = pruebas[-1]["prueba"]
    sufijo     = "safe" if args.safe_only else f"d{desde_real}_h{hasta_real}"
    tag        = f"_{args.tag}" if args.tag else ""
    salida     = RESULTS / f"{args.strategy}_{args.sheet}_{sufijo}{tag}.xlsx"
    timeout    = args.timeout if args.timeout > 0 else None
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sheet: {args.sheet}  |  Strategy: {args.strategy.upper()}")
    timeout_label = "sin límite" if timeout is None else f"{timeout}s"
    print(f"  Pruebas: {len(pruebas)}  |  ks: {args.ks}  |  timeout: {timeout_label}")
    print(f"  Salida: {salida.name}")
    print(f"{'='*60}\n")

    todas_filas = []

    for idx, prueba in enumerate(pruebas, start=1):
        segura = es_segura(prueba, args.sheet)
        etiqueta = "" if segura else " [!!riesgo]"
        print(f"[{idx:>3}/{len(pruebas)}] prueba #{prueba['prueba']:>3} "
              f"alc={prueba['alcance']:<25} mec={prueba['mecanismo']:<25}{etiqueta} ... ",
              end="", flush=True)

        ejecucion = ejecutar_prueba(prueba, args.sheet, args.strategy, args.ks, timeout)
        filas     = filas_de_resultado(prueba, args.sheet, args.strategy, ejecucion)
        todas_filas.extend(filas)

        estado = ejecucion["status"]
        t      = ejecucion.get("tiempo_total", 0)
        if estado == "ok":
            ks_ok = sorted(ejecucion["resultado"].keys())
            perdidas = [f"k={k}: φ={ejecucion['resultado'][k]['perdida']:.4f}" for k in ks_ok]
            print(f"OK ({t:.1f}s)  {' | '.join(perdidas)}")
        else:
            print(f"{estado.upper()} ({t:.1f}s)  {ejecucion.get('error', '')[:60]}")

        # Guardar parcial después de cada prueba
        pd.DataFrame(todas_filas).to_excel(salida, index=False)

    print(f"\n[OK] Resultados guardados en: {salida}")
    print(f"  Total filas: {len(todas_filas)}")

    ok      = sum(1 for f in todas_filas if f["status"] == "ok")
    timeout = sum(1 for f in todas_filas if f["status"] == "timeout")
    oom     = sum(1 for f in todas_filas if f["status"] == "oom")
    err     = sum(1 for f in todas_filas if f["status"] == "error")
    print(f"  ok={ok}  timeout={timeout}  oom={oom}  error={err}")


if __name__ == "__main__":
    main()
