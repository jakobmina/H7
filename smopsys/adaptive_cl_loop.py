"""
adaptive_cl_loop.py — Adaptive CL Loop with QuoreMindHP + H7 Algorithm

Integra la lógica bayesiana de BayesLogicHP (API real) con el controlador H7
para ajustar estimulación en tiempo real basado en análisis de alta precisión.

Mandato Metripléxico:
  d_symp = innovación O_n (movimiento conservativo del loop)
  d_metr = corrección H7 (relajación hacia el atractor de baja latencia)

Bridge:
  Cada tick escribe a cl1_session.sqlite (WAL) → leído por streamlit_monitor.py
"""

import cl
from pathlib import Path
import sys

# Adjust path to allow imports from dashboard
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dashboard.cl1_db import CL1Database
import time
import numpy as np
import math
from smopsys.quoremind_lite import BayesLogicLite, BayesLogicConfigLite, StatisticalAnalysisLite

RUN_FOR_SECONDS  = 30
TICKS_PER_SECOND = 1000
RUN_FOR_TICKS    = RUN_FOR_SECONDS * TICKS_PER_SECOND
PERCENTILES      = [0.001, 0.01, 0.1, 1, 99, 99.9, 99.99, 99.999]

# ─── Constante metripléctica ───────────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2


class H7AdaptiveController:
    """
    H7 Algorithm — Marco de decisión adaptativa para CL.

    Usa BayesLogicLite para calcular P(bottleneck | loop_duration).

    d_symp: Estimulación nominal conservativa
    d_metr: Reducción de complejidad para liberar carga cuando p_bottleneck > umbral
    """

    def __init__(self):
        config = BayesLogicConfigLite(
            epsilon=1e-15,
            high_entropy_threshold=0.8,
            high_coherence_threshold=0.6,
            action_threshold=0.5,
        )
        self.bayes = BayesLogicLite(config)

        # Acumuladores para análisis estadístico
        self._loop_durations: list = []
        self._spike_counts:   list = []

        # Parámetros de estimulación — ajustables en tiempo real
        self.stim_amp_ua: float = 1.5   # µA
        self.stim_dur_us: int   = 180   # µs

    # ── Lagrangiano Explícito (Regla 3.1) ──────────────────────────────────────
    def compute_lagrangian(self, loop_duration_us: float, spike_count: int):
        """
        L_symp: componente conservativa — continuidad del loop en tiempo real.
        L_metr: componente disipativa  — penalización por exceso de latencia.
        """
        target_us = 1_000_000.0 / TICKS_PER_SECOND  # 1000 µs para 1kHz
        L_symp =  target_us - loop_duration_us       # positivo = dentro de tiempo
        L_metr = -max(0.0, loop_duration_us - target_us)  # negativo si hay jitter
        return L_symp, L_metr

    # ── Análisis con BayesLogicLite ───────────────────────────────
    def analyze_tick(self, loop_duration_us: float, spike_count: int) -> float:
        """
        Usa calculate_posterior_probability(prior_a, prior_b, conditional_b_given_a)
        para estimar P(bottleneck | loop supera 1500µs).

        Mapeo metripléximo:
          prior_a             = P(bottleneck) a priori = 0.1
          prior_b             = P(loop excede umbral) = coherencia del sistema
          conditional_b_given_a = P(loop excede umbral | hay bottleneck)
        """
        self._loop_durations.append(loop_duration_us)
        self._spike_counts.append(float(spike_count))

        BOTTLENECK_THRESHOLD_US = 1_500.0

        # prior_a: P(bottleneck) base
        prior_a = 0.10

        # prior_b: si la media reciente sube, aumentamos la evidencia de congestión
        if len(self._loop_durations) > 10:
            recent_mean = float(np.mean(self._loop_durations[-10:]))
            coherence = min(1.0, recent_mean / BOTTLENECK_THRESHOLD_US)
        else:
            coherence = 0.3
        prior_b = float(coherence)

        # conditional_b_given_a: P(excede umbral | bottleneck real) = alta si este tick tardó mucho
        cond_b_given_a = 0.95 if loop_duration_us > BOTTLENECK_THRESHOLD_US else 0.05

        # ← API CORRECTA de BayesLogicLite
        prob_bottleneck = self.bayes.calculate_posterior_probability(
            prior_a        = prior_a,
            prior_b        = prior_b,
            conditional_b_given_a = cond_b_given_a,
        )
        return prob_bottleneck

    # ── Decisión H7 ────────────────────────────────────────────────────────────
    def decide(self, prob_bottleneck: float) -> str:
        """
        Regla H7: si P(bottleneck) > 0.80 → componente métrica activa (d_metr).
        De lo contrario, el sistema permanece en régimen conservativo (d_symp).
        """
        p = float(prob_bottleneck)
        if p > 0.80:
            print(f"[H7 d_metr] Bottleneck P={p:.3f} > 0.80 → reduciendo carga")
            # Reducir duración de pulso para liberar tiempo de cómputo
            self.stim_dur_us = max(100, self.stim_dur_us - 20)
            return "reduce_complexity"
        elif p < 0.15:
            # Recuperar parámetros nominales gradualmente (d_symp)
            self.stim_dur_us = min(180, self.stim_dur_us + 5)
            return "increase_precision"
        return "maintain"

    # ── Estadísticas Finales (API Lite StatisticalAnalysisLite) ──────────────
    def final_stats(self):
        """Retorna entropía de Shannon y estadísticas de los intervalos del loop."""
        if not self._loop_durations:
            return {}
        durations = self._loop_durations
        entropy = StatisticalAnalysisLite.shannon_entropy(
            [round(d / 100) for d in durations]   # discretizar para entropía simbólica
        )
        return {
            "mean_us":   float(np.mean(durations)),
            "std_us":    float(np.std(durations)),
            "min_us":    float(np.min(durations)),
            "max_us":    float(np.max(durations)),
            "entropy_hp": float(entropy),
        }


# ─── Loop Principal ────────────────────────────────────────────────────────────
def adaptive_loop_test():
    """Loop CL con análisis H7 + QuoreMindHP en cada tick."""

    loop_times_ns = np.empty(RUN_FOR_TICKS, dtype=np.int64)
    spike_count   = 0
    skipped_stims = 0
    h7            = H7AdaptiveController()

    print("🚀 Starting Adaptive CL Loop — QuoreMindHP + H7")
    print(f"   Target: {TICKS_PER_SECOND} Hz | {RUN_FOR_SECONDS}s | {RUN_FOR_TICKS} ticks")
    print("   Bridge: cl1_session.sqlite (WAL) → streamlit_monitor.py\n")

    db = CL1Database()
    db.new_session(TICKS_PER_SECOND, RUN_FOR_SECONDS)

    with cl.open() as neurons:
        for tick in neurons.loop(
            TICKS_PER_SECOND,
            stop_after_ticks=RUN_FOR_TICKS,
            ignore_jitter=True,
        ):
            t_now = time.monotonic_ns()
            loop_times_ns[tick.iteration] = t_now

            # ── Análisis Metripléxico por tick ──────────────────────────────
            loop_dur_us = None
            L_symp = L_metr = prob_b = None
            action = "maintain"

            if tick.iteration > 0:
                loop_dur_us = (t_now - loop_times_ns[tick.iteration - 1]) / 1_000.0
                spikes_now  = len(tick.analysis.spikes)

                L_symp, L_metr = h7.compute_lagrangian(loop_dur_us, spikes_now)
                prob_b         = h7.analyze_tick(loop_dur_us, spikes_now)
                action         = h7.decide(prob_b)

                if action != "maintain":
                    db.write_h7_event(
                        event_type  = action,
                        description = f"tick={tick.iteration} dur={loop_dur_us:.1f}µs spikes={spikes_now}",
                        value       = float(prob_b),
                    )

            # ── Estimulación Adaptativa ──────────────────────────────────────
            spikes_now  = len(tick.analysis.spikes)
            spike_count += spikes_now
            stim_fired  = False

            if spikes_now > 5:
                try:
                    neurons.stim(
                        cl.ChannelSet(27),
                        cl.StimDesign(
                            h7.stim_dur_us, -h7.stim_amp_ua,
                            h7.stim_dur_us,  h7.stim_amp_ua,
                        ),
                    )
                    stim_fired = True
                except Exception as e:
                    if "Skipped stim" in str(e):
                        skipped_stims += 1
                    else:
                        print(f"  Stim error: {e}")

            # ── Escribir tick a SQLite ────────────────────────────────────────
            db.write_tick(
                ts_ns           = t_now,
                loop_dur_us     = loop_dur_us,
                spike_count     = spikes_now,
                stim_fired      = stim_fired,
                prob_bottleneck = float(prob_b) if prob_b is not None else None,
                l_symp          = float(L_symp) if L_symp is not None else None,
                l_metr          = float(L_metr) if L_metr is not None else None,
                stim_dur_us     = h7.stim_dur_us,
                stim_amp_ua     = h7.stim_amp_ua,
            )

    # ─── Informe Final ─────────────────────────────────────────────────────────
    db.finalize_session(skipped_stims=skipped_stims, total_spikes=spike_count)
    db.close()

    intervals_us = np.diff(loop_times_ns) / 1_000.0
    percentiles  = [(p, np.percentile(intervals_us, p)) for p in PERCENTILES]
    stats        = h7.final_stats()

    print("\n" + "="*55)
    print("  VALIDATION SUMMARY — Adaptive CL + H7 + QuoreMindHP")
    print("="*55)
    print(f"  Duration      : {RUN_FOR_SECONDS}s @ {TICKS_PER_SECOND} Hz")
    print(f"  Total Spikes  : {spike_count} ({spike_count/RUN_FOR_SECONDS:.3f} spk/s)")
    print(f"  Skipped Stims : {skipped_stims}")
    print(f"  Mean Interval : {np.mean(intervals_us):.3f} µs")
    print(f"  Min / Max     : {np.min(intervals_us):.3f} / {np.max(intervals_us):.3f} µs")
    if stats:
        print(f"  Shannon Hₓ   : {stats['entropy_hp']:.6f} bits")
    print("\n  Percentile Distribution (µs):")
    for pct, val in percentiles:
        print(f"    {pct:>8.3f}th : {val:.3f}")


if __name__ == "__main__":
    adaptive_loop_test()
