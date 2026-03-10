"""
cl1_db.py — SQLite WAL Bridge entre el loop CL1 y el Streamlit Monitor

El loop CL1 (adaptive_cl_loop.py) escribe en tiempo real.
El monitor Streamlit (streamlit_monitor.py) lee sin bloquear.

Modo WAL (Write-Ahead Log) permite escritura y lectura simultáneas
desde distintos procesos sin contención. Ideal para 1000 Hz de datos.

Esquema:
  ticks   — un registro por tick: timestamp, duración, spike_count, prob_bottleneck
  spikes  — spikes individuales si se quiere granularidad fina (opcional)
  events  — eventos H7: correcciones, cambios de parámetro, triggers

Autor: Jacobo Tlacaelel Mina Rodriguez
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional
import threading

DB_PATH = Path(__file__).parent / "cl1_session.sqlite"

# ─── SQL Schema ────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS ticks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns           INTEGER NOT NULL,           -- tiempo monotónico en ns
    loop_dur_us     REAL,                       -- duración del tick en µs
    spike_count     INTEGER DEFAULT 0,          -- spikes en este tick
    stim_fired      INTEGER DEFAULT 0,          -- 1 si se disparó estimulación
    prob_bottleneck REAL,                       -- P(bottleneck) bayesiana
    l_symp          REAL,                       -- Lagrangiano conservativo
    l_metr          REAL,                       -- Lagrangiano disipativo
    stim_dur_us     INTEGER,                    -- parámetro actual de estimulación
    stim_amp_ua     REAL                        -- amplitud actual
);

CREATE TABLE IF NOT EXISTS h7_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns       INTEGER NOT NULL,
    event_type  TEXT NOT NULL,          -- 'bottleneck', 'correction', 'maintain'
    description TEXT,
    value       REAL
);

CREATE TABLE IF NOT EXISTS session_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class CL1Database:
    """
    Interfaz de escritura para el loop CL1.
    Thread-safe: usa una conexión por hilo.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Retorna conexión thread-local (una por hilo)."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Crea las tablas si no existen y activa WAL."""
        conn = self._get_conn()
        conn.executescript(SCHEMA)
        conn.commit()

    def new_session(self, ticks_per_second: int, run_for_seconds: int):
        """Registra metadatos de la sesión actual."""
        conn = self._get_conn()
        conn.execute("DELETE FROM ticks")
        conn.execute("DELETE FROM h7_events")
        conn.execute("""
            INSERT OR REPLACE INTO session_meta(key, value) VALUES
                ('start_ts', ?), ('tps', ?), ('duration_s', ?), ('status', 'running')
        """, (str(time.monotonic_ns()), str(ticks_per_second), str(run_for_seconds)))
        conn.commit()

    def write_tick(
        self,
        ts_ns: int,
        loop_dur_us: Optional[float],
        spike_count: int,
        stim_fired: bool,
        prob_bottleneck: Optional[float],
        l_symp: Optional[float],
        l_metr: Optional[float],
        stim_dur_us: int,
        stim_amp_ua: float,
    ):
        """Inserta un tick. Operación rápida (<50µs) para no bloquear el loop."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO ticks
              (ts_ns, loop_dur_us, spike_count, stim_fired,
               prob_bottleneck, l_symp, l_metr, stim_dur_us, stim_amp_ua)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            ts_ns,
            loop_dur_us,
            spike_count,
            1 if stim_fired else 0,
            prob_bottleneck,
            l_symp,
            l_metr,
            stim_dur_us,
            stim_amp_ua,
        ))
        conn.commit()

    def write_h7_event(self, event_type: str, description: str = "", value: float = 0.0):
        """Registra un evento H7 (bottleneck detectado, corrección aplicada, etc.)."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO h7_events (ts_ns, event_type, description, value)
            VALUES (?, ?, ?, ?)
        """, (time.monotonic_ns(), event_type, description, value))
        conn.commit()

    def finalize_session(self, skipped_stims: int, total_spikes: int):
        """Marca la sesión como completada."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO session_meta(key, value) VALUES
                ('status', 'done'),
                ('skipped_stims', ?),
                ('total_spikes', ?)
        """, (str(skipped_stims), str(total_spikes)))
        conn.commit()

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()


class CL1Reader:
    """
    Interfaz de lectura para el Streamlit Monitor.
    Solo lectura, sin contención con el loop.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get_recent_ticks(self, n: int = 500):
        """Retorna los últimos N ticks como lista de dicts."""
        if not self.db_path.exists():
            return []
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM ticks ORDER BY id DESC LIMIT ?
            """, (n,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_summary(self) -> dict:
        """Estadísticas agregadas de la sesión en curso."""
        if not self.db_path.exists():
            return {}
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                        AS total_ticks,
                    SUM(spike_count)                AS total_spikes,
                    SUM(stim_fired)                 AS stim_count,
                    AVG(loop_dur_us)                AS mean_dur_us,
                    MIN(loop_dur_us)                AS min_dur_us,
                    MAX(loop_dur_us)                AS max_dur_us,
                    AVG(prob_bottleneck)            AS mean_prob_bottleneck,
                    SUM(CASE WHEN stim_fired=0 AND spike_count>5 THEN 1 ELSE 0 END) AS skipped_stims
                FROM ticks
            """).fetchone()
            meta = dict(conn.execute("SELECT key, value FROM session_meta").fetchall() or [])
            events = conn.execute("""
                SELECT event_type, COUNT(*) as n FROM h7_events GROUP BY event_type
            """).fetchall()
        result = dict(row) if row else {}
        result["meta"] = meta
        result["events"] = {r["event_type"]: r["n"] for r in events}
        return result

    def get_h7_events(self, n: int = 50):
        """Últimos N eventos H7."""
        if not self.db_path.exists():
            return []
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM h7_events ORDER BY id DESC LIMIT ?
            """, (n,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def session_status(self) -> str:
        """Retorna 'running', 'done', o 'no_session'."""
        if not self.db_path.exists():
            return "no_session"
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM session_meta WHERE key='status'"
                ).fetchone()
            return row["value"] if row else "no_session"
        except Exception:
            return "no_session"
