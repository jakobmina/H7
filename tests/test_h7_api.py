"""
tests/test_h7_api.py — Pytest suite for the H7 FastAPI backend
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from h7_api import app

client = TestClient(app)


# ── /api/h7/matrix ────────────────────────────────────────────────────────────
class TestMatrix:
    def test_returns_20_rows(self):
        r = client.get("/api/h7/matrix")
        assert r.status_code == 200
        data = r.json()
        assert len(data["rows"]) == 20   # 4 states × 5 violations

    def test_row_has_required_fields(self):
        r = client.get("/api/h7/matrix")
        row = r.json()["rows"][0]
        for field in ("state","violation","n","sv","psi","t","fw","bw","label","key"):
            assert field in row, f"Missing field: {field}"

    def test_label_values_valid(self):
        r = client.get("/api/h7/matrix")
        valid = {"constructive","equilibrium","destructive"}
        for row in r.json()["rows"]:
            assert row["label"] in valid

    def test_ternary_consistent_with_psi(self):
        r = client.get("/api/h7/matrix")
        for row in r.json()["rows"]:
            p, t = row["psi"], row["t"]
            if p > 0.25:
                assert t == 1
            elif p < -0.25:
                assert t == -1
            else:
                assert t == 0

    def test_summary_sums_to_20(self):
        r = client.get("/api/h7/matrix")
        s = r.json()["summary"]
        assert sum(s.values()) == 20

    def test_source_is_python(self):
        r = client.get("/api/h7/matrix")
        assert r.json()["source"] == "python-backend"


# ── /api/h7/oracle ────────────────────────────────────────────────────────────
class TestOracle:
    def test_returns_200(self):
        r = client.get("/api/h7/oracle")
        assert r.status_code == 200

    def test_has_hidden_symmetry(self):
        r = client.get("/api/h7/oracle")
        data = r.json()
        assert "hidden_symmetry" in data
        assert isinstance(data["hidden_symmetry"], int)

    def test_lagrangians_for_all_momenta(self):
        r = client.get("/api/h7/oracle")
        lags = r.json()["lagrangians"]
        for p in ["1","2","3","4","5","6"]:
            assert p in lags
            entry = lags[p]
            assert "L_symp" in entry and "L_metr" in entry

    def test_metriplectic_rule_L_metr_negative(self):
        """Regla 1.2: L_metr must be ≤ 0 (dissipative component)."""
        r = client.get("/api/h7/oracle")
        for p, d in r.json()["lagrangians"].items():
            assert d["L_metr"] <= 0, f"L_metr > 0 for p={p}: {d['L_metr']}"

    def test_h7_pairing_sums_to_7(self):
        """H7 Conservation: state + partner = 7."""
        r = client.get("/api/h7/oracle")
        for state, partner in r.json()["h7_pairing"].items():
            assert int(state) + partner == 7, f"Pairing broken: {state}↔{partner}"


# ── /api/h7/status ────────────────────────────────────────────────────────────
class TestStatus:
    def test_returns_200(self):
        r = client.get("/api/h7/status")
        assert r.status_code == 200

    def test_has_lagrangian_fields(self):
        r = client.get("/api/h7/status")
        for field in ("L_symp","L_metr","O_n","stability","phi","tick"):
            assert field in r.json(), f"Missing field: {field}"

    def test_stability_in_range(self):
        r = client.get("/api/h7/status")
        s = r.json()["stability"]
        assert 0.0 <= s <= 1.0, f"Stability out of [0,1]: {s}"

    def test_L_metr_is_nonpositive(self):
        """Regla 1.2: dissipative component must be ≤ 0."""
        r = client.get("/api/h7/status")
        assert r.json()["L_metr"] <= 0
