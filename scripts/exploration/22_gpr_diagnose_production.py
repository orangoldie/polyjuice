"""Diagnostic — compare sibling vs production GPR coefficients, and verify
that production coefs are consistent with sibling (z=8.79 on T=10).

If production has flipped or shrunk the GPR contrast vs sibling, that would
explain why GPR underperforms on the 2026 holdout despite the strong z-stat
we measured on the sibling.
"""
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIB = os.path.join(ROOT, "data/models/baseline_logistic_gpr")
PROD = os.path.join(SIB, "production")
OUT_TXT = os.path.join(ROOT, "data", "exploration",
                       "22_gpr_diagnose_production.txt")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


_fh = open(OUT_TXT, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _fh)


def coefs(payload):
    feats = payload["feature_names"]
    arr = payload["model"].coef_[0]
    return dict(zip(feats, arr))


for T in (10, 15, 20, 25):
    sib = pickle.load(open(os.path.join(SIB, f"checkpoint_{T}/model.pkl"), "rb"))
    prod = pickle.load(open(os.path.join(PROD, f"checkpoint_{T}/model.pkl"), "rb"))
    cs = coefs(sib)
    cp = coefs(prod)
    print(f"=== T={T} ===  sibling C={sib['model'].C_[0]:g}  production C={prod['model'].C_[0]:g}")
    print(f"{'feature':<14} {'sibling':>10} {'production':>12} {'delta':>10}")
    for f in ("gpr_team", "gpr_opp_team", "has_gpr"):
        print(f"{f:<14} {cs[f]:>+10.4f} {cp[f]:>+12.4f} {cp[f]-cs[f]:>+10.4f}")
    contrast_s = cs["gpr_team"] - cs["gpr_opp_team"]
    contrast_p = cp["gpr_team"] - cp["gpr_opp_team"]
    print(f"{'contrast':<14} {contrast_s:>+10.4f} {contrast_p:>+12.4f} {contrast_p-contrast_s:>+10.4f}")
    print(f"  mu_team={prod['mu_gpr']['gpr_team']:.2f}  mu_opp={prod['mu_gpr']['gpr_opp_team']:.2f}")
    # Compare scaler stats too — production scaler trains on 2020-2025 (incl 2025);
    # sibling trains on 2020-2024. Different rated-row distributions => different
    # scale factors => same raw rating gap maps to different standardized contrast.
    sib_scale = dict(zip(sib["numeric_cols"], sib["scaler"].scale_))
    prod_scale = dict(zip(prod["numeric_cols"], prod["scaler"].scale_))
    print(f"  scaler.scale_  sibling: gpr_team={sib_scale['gpr_team']:.2f}  "
          f"opp={sib_scale['gpr_opp_team']:.2f}")
    print(f"  scaler.scale_  prod:    gpr_team={prod_scale['gpr_team']:.2f}  "
          f"opp={prod_scale['gpr_opp_team']:.2f}")
    print()

sys.stdout = sys.__stdout__
_fh.close()
print(f"Wrote {OUT_TXT}")
