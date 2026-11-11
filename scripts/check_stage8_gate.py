#!/usr/bin/env python
"""Fail-closed gate for stage 8 (reserved sets). Opens nothing; it only reads the stage-7 review JSON.

Usage: python scripts/check_stage8_gate.py path/to/revision_stage7_*.json
Exit 0 only if EVERY condition holds; otherwise prints each unmet condition and exits 1.
Conditions (docs/project/hoja_de_ruta_fase1_original.md sec. 11 + revision_stage7 fields):
  pair.both_computational_locks_verified, pair.external_review != 'pending', pair.stage8_authorized,
  pair.reserved_access == [], and for each regime: no failures, offline_recovery_present (user-run on a
  different Colab runtime), no pending human-review items.
"""
import json, sys

def main(path):
    r = json.load(open(path, encoding="utf-8")); bad = []
    p = r["pair"]
    if not p.get("both_computational_locks_verified"): bad.append("locks computacionales no verificados")
    if p.get("external_review") == "pending": bad.append("revisión externa/humana pendiente")
    if not p.get("stage8_authorized"): bad.append("stage8_authorized=false")
    if p.get("reserved_access"): bad.append("reserved_access no vacío: %r" % p["reserved_access"])
    for name, g in r["regimes"].items():
        if g.get("failures"): bad.append(f"{name}: fallos {g['failures']}")
        if not g.get("offline_recovery_present"): bad.append(f"{name}: falta recover offline en otro runtime de Colab")
        for item in g.get("pending", []): bad.append(f"{name}: pendiente: {item}")
    if bad:
        print("ETAPA 8 BLOQUEADA:"); [print(" -", b) for b in dict.fromkeys(bad)]; return 1
    print("Compuerta abierta: revisar manualmente el cierre antes de ejecutar notebooks/08."); return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
