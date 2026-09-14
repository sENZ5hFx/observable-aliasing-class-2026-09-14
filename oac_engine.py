#!/usr/bin/env python3
"""Observable-Aliasing Class (OAC) scoring engine — session 2026-09-14.

Research-stage classification. Not a physical discovery, not peer review,
not a patent. Scores are a heuristic over sourced gaps, not measurements
of nature.

OAC: independent physical quantities are collapsed by a rank-deficient
observation operator into a single reported number or named object.
The 'crisis' is often the alias, not new physics.

Dual of PPC (2026-09-08): PPC adds hidden degrees of freedom that absorb
residuals. OAC has too few independent observables.

Identifiability is prior art (Koopmans 1949; Wyithe 2010; Wang & Shan 2026).
This engine does not claim that math. It asks which remaining cross-realm
gaps, after nineteen archived classes, are the same operator.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

SESSION_ID = "2026-09-14-oac-observable-aliasing"
TIMESTAMP = "2026-09-14T13:40:00-04:00"

Status = Literal["STRUCTURAL_GAP", "CALIBRATION", "PENALIZED", "KILLED", "ZEROED", "CLOSED"]


@dataclass
class Instance:
    id: str
    realm: str
    title: str
    observable: str
    aliased_a: str
    aliased_b: str
    source: str
    date: str
    rank_deficiency: float
    home_field_named: float
    prior_overlap: float
    discriminator_runnable: float
    freshness: float
    crisis_was_alias: float
    notes: str = ""
    prior_owner: str = ""


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_oac_instance(inst: Instance) -> dict:
    """Score a candidate against OAC gates after archive subtraction.

    Inputs: an Instance with 0–1 features and archive-overlap.
    Outputs: dict with score, status, reasons, promoted.

    Gates:
      - prior_overlap >= 0.85 → ZEROED (archive already owns it)
      - notes mention extra hidden DOF → KILLED (that is PPC)
      - home_field_named >= 0.7 → CALIBRATION (do not let Wang & Shan win)
      - CLOSED prefix → CLOSED calibration of the operator
      - prior_overlap >= 0.45 → PENALIZED
    """
    reasons: list[str] = []
    status: Status = "STRUCTURAL_GAP"

    if inst.prior_overlap >= 0.85:
        return _pack(inst, 0.0, "ZEROED", [f"archive owns this: {inst.prior_owner}"], False)

    if "hidden field" in inst.notes.lower() or "extra dof" in inst.notes.lower():
        return _pack(inst, 0.0, "KILLED", ["fix is extra hidden DOF → PPC, not OAC"], False)

    ident = (
        0.34 * inst.rank_deficiency
        + 0.22 * inst.crisis_was_alias
        + 0.18 * inst.discriminator_runnable
        + 0.16 * inst.freshness
        + 0.10 * (1.0 - inst.home_field_named)
    )
    ident *= 1.0 - 0.75 * inst.prior_overlap

    if inst.home_field_named >= 0.7:
        ident *= 0.42
        status = "CALIBRATION"
        reasons.append("home field already named this degeneracy; calibration only")

    raw = 100.0 * clamp(ident)

    if inst.prior_overlap >= 0.45:
        raw *= 0.55
        status = "PENALIZED"
        reasons.append(f"partial overlap with {inst.prior_owner}")

    if raw < 8 and status == "STRUCTURAL_GAP":
        status = "PENALIZED"
        reasons.append("below structural threshold")

    if inst.id.startswith("CLOSED"):
        status = "CLOSED"
        reasons.append("finding closed; retained as operator calibration")

    promoted = status == "STRUCTURAL_GAP" and raw >= 28.0
    return _pack(inst, round(raw, 2), status, reasons, promoted)


def _pack(inst: Instance, score: float, status: Status, reasons: list[str], promoted: bool) -> dict:
    d = asdict(inst)
    d.update({"score": score, "status": status, "reasons": reasons, "promoted": promoted})
    return d


def degeneracy_manifold_width(
    reported_crisis: float,
    unmixed_low: float,
    unmixed_high: float,
    units: str,
) -> dict:
    """Estimate how much of a named crisis was alias vs remainder.

    alias_share = |reported - midpoint(unmixed)| / (|Δ| + width).
    Heuristic only — not a measurement of nature.
    """
    if not math.isfinite(reported_crisis):
        raise ValueError("reported_crisis must be finite")
    if unmixed_high < unmixed_low:
        raise ValueError("unmixed_high must be >= unmixed_low")
    mid = 0.5 * (unmixed_low + unmixed_high)
    delta = abs(reported_crisis - mid)
    width = unmixed_high - unmixed_low
    denom = delta + width if (delta + width) > 0 else 1e-12
    alias_share = delta / denom
    if alias_share >= 0.6:
        interpretation = "most of the named crisis was alias"
    elif alias_share >= 0.3:
        interpretation = "unmixing leaves a comparable remainder"
    else:
        interpretation = "crisis mostly survives unmixing"
    return {
        "units": units,
        "reported": reported_crisis,
        "unmixed_interval": [unmixed_low, unmixed_high],
        "point_shift": round(delta, 6),
        "remaining_width": round(width, 6),
        "alias_share": round(alias_share, 4),
        "remainder_share": round(width / denom, 4),
        "interpretation": interpretation,
    }


def split_named_home_field(rows: list[dict]) -> dict:
    """Do not let a home-field-named calibration outrank a STRUCTURAL_GAP."""
    gaps = sorted([r for r in rows if r["status"] == "STRUCTURAL_GAP"], key=lambda r: -r["score"])
    cals = sorted([r for r in rows if r["status"] == "CALIBRATION"], key=lambda r: -r["score"])
    primary = gaps[0] if gaps else None
    mode = "structural_gap_holds"
    if primary and cals and cals[0]["score"] > primary["score"]:
        mode = "calibration_would_have_won_without_split"
    return {
        "mode": mode,
        "primary": primary["id"] if primary else None,
        "primary_score": primary["score"] if primary else None,
        "runner_up": gaps[1]["id"] if len(gaps) > 1 else None,
        "calibration_top": cals[0]["id"] if cals else None,
        "n_structural": len(gaps),
        "n_calibration": len(cals),
    }


def split_realm_monopoly(ranking: list[dict]) -> dict:
    """If one realm holds primary and runner-up, promote highest other-realm gap as co-primary."""
    gaps = [r for r in ranking if r["status"] == "STRUCTURAL_GAP"]
    if len(gaps) < 2:
        return {"mode": "insufficient", "co_primary": None}
    if gaps[1]["realm"] == gaps[0]["realm"]:
        other = [g for g in gaps if g["realm"] != gaps[0]["realm"]]
        co = other[0] if other else None
        mode = "co_primary_forced"
    else:
        co = None
        mode = "realms_already_split"
    return {
        "mode": mode,
        "primary": gaps[0]["id"],
        "primary_realm": gaps[0]["realm"],
        "runner_up": gaps[1]["id"],
        "co_primary": co["id"] if co else None,
        "co_primary_realm": co["realm"] if co else None,
        "co_primary_score": co["score"] if co else None,
    }


def sha256_obj(obj: dict) -> str:
    blob = json.dumps(obj, indent=2, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
