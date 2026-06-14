"""
runner.py — lógica de ejecución, carga de datos y exportación.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT          = Path(__file__).resolve().parent
WORKER_QNODES = ROOT / "logica" / "_worker_qnodes.py"
WORKER_GEOMIP = ROOT / "logica" / "_worker_geomip_recursivo.py"
ABECEDARY     = "ABCDEFGHIJKLMNOPQRSTUVWXY"


# ── carga de datos ────────────────────────────────────────────────────────────

def letras_a_binario(texto: str, n: int) -> str:
    bits = ["0"] * n
    for c in texto.upper():
        p = ABECEDARY.find(c)
        if 0 <= p < n:
            bits[p] = "1"
    return "".join(bits)


def parse_tpm_text(text: str) -> np.ndarray:
    """Parsea TPM desde texto multilinea (espacio o coma). Lanza ValueError."""
    rows = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            vals = [float(v) for v in line.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"Valor no numérico en la línea {i}.")
        if rows and len(vals) != len(rows[0]):
            raise ValueError(
                f"Línea {i}: {len(vals)} columnas, se esperaban {len(rows[0])}.")
        rows.append(vals)
    if not rows:
        raise ValueError("La TPM está vacía.")
    return np.array(rows, dtype=float)


def load_tpm_csv(path: str) -> np.ndarray:
    """
    Carga TPM desde CSV. Acepta cualquier archivo .csv con formato:
      - Valores numéricos separados por comas (0s y 1s, o probabilidades)
      - 2^N filas × N columnas, sin columna índice
      - Con o sin fila de encabezado (se detecta automáticamente)
    Lanza ValueError con mensaje descriptivo si el formato es incorrecto.
    """
    import pandas as pd
    try:
        # Intento sin header
        df = pd.read_csv(path, header=None)
        try:
            arr = df.values.astype(float)
        except (ValueError, TypeError):
            # Primera fila es texto → usarla como header y descartar
            df = pd.read_csv(path, header=0)
            arr = df.values.astype(float)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}")

    if arr.ndim != 2:
        raise ValueError("El archivo no contiene una matriz 2D válida.")
    if not np.isfinite(arr).all():
        raise ValueError(
            "El CSV contiene valores no numéricos o celdas vacías. "
            "Asegúrate de que todas las celdas tengan valores numéricos."
        )
    return arr


def count_excel_rows(path: str, sheet: str) -> int:
    import pandas as pd
    df = pd.read_excel(path, sheet_name=sheet, skiprows=4, header=0)
    df = df.dropna(subset=[df.columns[0]])
    return len(df)


def load_excel_rows(path: str, sheet: str, desde: int, hasta: int) -> list:
    import pandas as pd
    df = pd.read_excel(path, sheet_name=sheet, skiprows=4, header=0)
    df = df.dropna(subset=[df.columns[0]])
    df.columns = ["prueba", "alcance", "mecanismo"] + list(df.columns[3:])
    rows = []
    for _, row in df.iterrows():
        num = int(row["prueba"])
        if num < desde or num > hasta:
            continue
        rows.append({
            "prueba":    num,
            "alcance":   str(row["alcance"]).strip(),
            "mecanismo": str(row["mecanismo"]).strip(),
        })
    return rows


def generate_tpm(n: int, semilla: int) -> np.ndarray:
    np.random.seed(semilla)
    return np.random.randint(2, size=(2**n, n), dtype=np.int8).astype(float)


def export_results(results: list, path: str) -> None:
    import pandas as pd
    rows = []
    for res in results:
        p = res.params
        for k_str, data in res.raw.items():
            rows.append({
                "algoritmo":      p.algo.upper(),
                "prueba":         p.prueba_num,
                "alcance":        p.alcance_label,
                "mecanismo":      p.mecanismo_label,
                "k":              int(k_str),
                "perdida":        data.get("perdida", 0.0),
                "tiempo_k":       data.get("tiempo", 0.0),
                "tiempo_total":   res.elapsed,
                "estado_inicial": p.estado_inicial,
                "condicion":      p.condicion,
                "particion":      data.get("particion", ""),
            })
    pd.DataFrame(rows).to_excel(path, index=False)


# ── parámetros y resultado ────────────────────────────────────────────────────

class ExecutionParams:
    def __init__(
        self,
        algo: str,
        tpm: np.ndarray,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        ks: list,
        semilla: int = 73,
        prueba_num: int = 1,
        alcance_label: str = "",
        mecanismo_label: str = "",
    ):
        self.algo            = algo
        self.tpm             = tpm
        self.estado_inicial  = estado_inicial
        self.condicion       = condicion
        self.alcance         = alcance
        self.mecanismo       = mecanismo
        self.ks              = ks
        self.semilla         = semilla
        self.prueba_num      = prueba_num
        self.alcance_label   = alcance_label or alcance
        self.mecanismo_label = mecanismo_label or mecanismo

    def to_payload(self) -> dict:
        return {
            "N":              len(self.estado_inicial),
            "estado_inicial": self.estado_inicial,
            "condicion_bin":  self.condicion,
            "alcance_bin":    self.alcance,
            "mecanismo_bin":  self.mecanismo,
            "ks":             self.ks,
            "semilla":        self.semilla,
            "tpm_data":       self.tpm.tolist(),
        }


class RunResult:
    def __init__(self, params: ExecutionParams, raw: dict, elapsed: float):
        self.params  = params
        self.raw     = raw      # {k_str: {particion, perdida, tiempo}}
        self.elapsed = elapsed


# ── runner ────────────────────────────────────────────────────────────────────

class Runner:
    """Un slot de ejecución — un subprocess a la vez."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._proc:   Optional[subprocess.Popen]  = None
        self._lock    = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def execute(
        self,
        params: ExecutionParams,
        on_tick:   Callable[[float], None],
        on_result: Callable[["RunResult"], None],
        on_error:  Callable[[str], None],
    ) -> None:
        if self.running:
            return

        def _worker():
            import os, tempfile
            script  = str(WORKER_QNODES if params.algo == "qnodes" else WORKER_GEOMIP)
            payload = json.dumps(params.to_payload(), ensure_ascii=False)
            t0      = time.time()

            # Windows cmd-line limit is 32767 chars; use a temp file for large payloads
            tmp_path = None
            if len(payload) > 4096:
                fd, tmp_path = tempfile.mkstemp(suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                argv1 = f"@{tmp_path}"
            else:
                argv1 = payload

            try:
                with self._lock:
                    self._proc = subprocess.Popen(
                        [sys.executable, script, argv1],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                    )
                proc = self._proc

                while proc.poll() is None:
                    time.sleep(0.1)
                    on_tick(time.time() - t0)

                stdout, stderr = proc.communicate()
                elapsed = time.time() - t0

                if proc.returncode != 0:
                    lines = [l for l in stderr.strip().splitlines() if l.strip()]
                    on_error(lines[-1] if lines else "Error desconocido.")
                    return

                json_lines = [l for l in stdout.strip().splitlines() if l.strip()]
                if not json_lines:
                    on_error("El worker no produjo salida.")
                    return
                raw = json.loads(json_lines[-1])
                on_result(RunResult(params, raw, elapsed))

            except Exception as exc:
                on_error(str(exc))
            finally:
                with self._lock:
                    self._proc = None
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
