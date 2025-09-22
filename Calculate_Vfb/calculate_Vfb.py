# -*- coding: utf-8 -*-
"""
calculate_Vfb.py  (Cl3 단일/Cl3+Cl4 둘 다 자동 대응)
- Q–V, C–V
- Vfb (실측/이론), CV 변곡
- Vt (이론) = Vfb + 2φF + |Qd|/Cox
- Vt (QV) = 첫 번째 V ≥ Vfb 에서 psi_s_signed(V) = 2φF  (psi_s 부호 복원)
- 보기 좋은 요약 vfb_values.csv + pretty(md/txt), 그래프 저장
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="QV/CV & Vfb/Vt from nextnano int_space_charge_dens.dat")
parser.add_argument("--file", "-f", type=Path, default=Path("int_space_charge_dens.dat"))
parser.add_argument("--out", "-o", type=Path, default=Path("out_cv_qv"))
args = parser.parse_args()
FILE: Path = args.file
OUT:  Path = args.out
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Params (필요시 수정)
# -----------------------------
NA_cm3        = 1e17
ND_gate_cm3   = 1e20
ni_cm3        = 1e10
T_K           = 300.0
tox_nm        = 8.0
eps_ox_rel    = 3.9
Qox_C_per_cm2 = 0.0

# -----------------------------
# Constants
# -----------------------------
q   = 1.602176634e-19
kB  = 1.380649e-23
eps0_F_per_cm = 8.8541878128e-14

# -----------------------------
# Helpers
# -----------------------------
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=0, comment="#")
    except Exception:
        df = None
    if df is None or df.empty or not any(str(c).startswith("sweep") for c in df.columns):
        raw = pd.read_csv(path, sep=r"\s+", engine="python", header=None, comment="#")
        if raw.shape[1] >= 6:
            raw = raw.iloc[:, :6]
            raw.columns = ["sweep_index","Cl_1","Cl_2","Cl_3","Cl_4","Total"]
        elif raw.shape[1] == 5:
            raw.columns = ["sweep_index","Cl_1","Cl_2","Cl_3","Total"]
        else:
            cols = ["sweep_index","Cl_1","Cl_2","Cl_3"]
            raw.columns = cols[:raw.shape[1]]
        if isinstance(raw.iloc[0,0], str) and "sweep" in raw.iloc[0,0].lower():
            raw = raw.iloc[1:].reset_index(drop=True)
        df = raw
    ren = {}
    for c in df.columns:
        s = str(c).strip()
        if s.startswith("sweep"): ren[c] = "V"
        elif "Cl_1" in s or "Cl1" in s: ren[c] = "Cl1"
        elif "Cl_2" in s or "Cl2" in s: ren[c] = "Cl2"
        elif "Cl_3" in s or "Cl3" in s: ren[c] = "Cl3"
        elif "Cl_4" in s or "Cl4" in s: ren[c] = "Cl4"
        elif "Total" in s:              ren[c] = "Total"
    df = df.rename(columns=ren)
    for k in [c for c in ["V","Cl1","Cl2","Cl3","Cl4","Total"] if c in df.columns]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df.dropna(subset=["V","Cl3"]).sort_values("V").reset_index(drop=True)

def central_diff(x, y):
    x = np.asarray(x); y = np.asarray(y)
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dy[0]    = (y[1] - y[0])     / (x[1] - x[0])
    dy[-1]   = (y[-1]- y[-2])    / (x[-1]- x[-2])
    return dy

def zero_cross_x(x, y):
    for i in range(len(x)-1):
        if y[i] == 0: return float(x[i])
        if y[i]*y[i+1] < 0:
            r = abs(y[i])/(abs(y[i])+abs(y[i+1]))
            return float(x[i] + r*(x[i+1]-x[i]))
    return None

def inflection_x(x, f):
    df1 = central_diff(x, f)
    df2 = central_diff(x, df1)
    for i in range(len(x)-1):
        a, b = df2[i], df2[i+1]
        if a == 0: return float(x[i])
        if a*b < 0:
            r = abs(a)/(abs(a)+abs(b))
            return float(x[i] + r*(x[i+1]-x[i]))
    return None

# -----------------------------
# Load & basic quantities
# -----------------------------
df = load_table(FILE)

# Ns: (새 포맷은 Cl3 자체가 Ns, 옛 포맷은 Cl3+Cl4)
if "Cl4" in df.columns and df["Cl4"].notna().any():
    df["Ns_cm2"] = df["Cl3"] + df["Cl4"]
else:
    df["Ns_cm2"] = df["Cl3"]

df["Qs_C_per_cm2"] = q * df["Ns_cm2"]

V = df["V"].to_numpy()
Q = df["Qs_C_per_cm2"].to_numpy()
C = central_diff(V, Q)
df["C_F_per_cm2"] = C

Cox = eps_ox_rel * eps0_F_per_cm / (tox_nm * 1e-7)
df["C_over_Cox"] = df["C_F_per_cm2"] / Cox

# Vfb (실측/변곡) & theory
Vfb_meas  = zero_cross_x(V, Q)
V_inflect = inflection_x(V, C)

phiF_p     = (kB*T_K/q) * np.log(NA_cm3/ni_cm3)
phiF_nplus = (kB*T_K/q) * np.log(ND_gate_cm3/ni_cm3)
Phi_ms     = -phiF_nplus - phiF_p
Vfb_th     = Phi_ms - Qox_C_per_cm2 / Cox

# -----------------------------
# Threshold voltages
# -----------------------------
eps_si = 11.7 * eps0_F_per_cm

# (A) Theory
Qd_2phiF  = -np.sqrt(4.0 * q * eps_si * NA_cm3 * phiF_p)   # [C/cm^2] (음수)
Vt_theory = Vfb_th + 2.0*phiF_p + abs(Qd_2phiF)/Cox

# (B) QV-based with SIGNED psi_s
#     psi_s_signed = sgn(-Q)*Q^2/(2 q eps_si NA)
psi_s_signed = np.sign(-Q) * (Q**2) / (2.0 * q * eps_si * NA_cm3)  # [V]
target = 2.0 * phiF_p

#     교차는 V >= Vfb_meas 구간에서만 찾음 (공핍/반전 쪽 해)
mask = V >= (Vfb_meas if Vfb_meas is not None else V.min())
V_sel = V[mask]; psi_sel = psi_s_signed[mask]
Vt_qv = zero_cross_x(V_sel, psi_sel - target)

# -----------------------------
# Save tables
# -----------------------------
df.to_csv(OUT / "qv_cv_table_full.csv", index=False)

# C≈Cox 전압(참고)
idx_Ceq = int(np.argmin(np.abs(df["C_F_per_cm2"].to_numpy() - Cox)))
V_CeqCox = float(df.loc[idx_Ceq, "V"])

# -----------------------------
# Summaries (pretty)
# -----------------------------
def fmt_value(value, unit):
    if value is None: return ""
    if unit == "V": return f"{value:.6f}"
    if unit in ("F/cm^2","C/cm^2"): return f"{value:.3e}"
    if unit == "cm^-3": return f"{value:.3e}"
    if unit in ("K","nm","-"):
        return f"{value:.3f}" if isinstance(value,float) else str(value)
    try: return f"{float(value):.6g}"
    except: return str(value)

rows = [
    ("Vfb_measured (Qs=0)",            Vfb_meas,     "V",       "Q–V에서 Qs=0 교차(선형보간)"),
    ("CV inflection voltage",          V_inflect,    "V",       "C(V)에서 d²C/dV²=0"),
    ("V where C≈Cox",                  V_CeqCox,     "V",       "C가 Cox에 가장 가까운 전압"),
    ("Vfb_theory (Phi_ms - Qox/Cox)",  Vfb_th,       "V",       "이론식"),
    ("Vt_theory (Vfb+2φF+|Qd|/Cox)",   Vt_theory,    "V",       "교과서 정의"),
    ("Vt_from_QV (psi_s=2φF)",         Vt_qv,        "V",       "SIGNED psi_s, V≥Vfb에서 교차"),
    ("Qd(2φF)",                        Qd_2phiF,     "C/cm^2",  "임계 공핍전하(시트)"),
    ("phiF(p)",                        phiF_p,       "V",       "φF(p) = (kT/q) ln(NA/ni)"),
    ("phiF(n+)",                       phiF_nplus,   "V",       "φF(n+) = (kT/q) ln(ND/ni)"),
    ("Phi_ms",                         Phi_ms,       "V",       "≈ -φF(n+) - φF(p)"),
    ("Cox",                            Cox,          "F/cm^2",  "εox/tox"),
    ("Qox",                            Qox_C_per_cm2,"C/cm^2",  "입력 고정전하"),
    ("NA (p-sub)",                     NA_cm3,       "cm^-3",   ""),
    ("ND_gate (n+ poly)",              ND_gate_cm3,  "cm^-3",   ""),
    ("ni",                             ni_cm3,       "cm^-3",   ""),
    ("T",                              T_K,          "K",       ""),
    ("tox",                            tox_nm,       "nm",      ""),
    ("eps_ox_rel",                     eps_ox_rel,   "-",       ""),
]
pretty_rows = [{"Metric": m, "Value": fmt_value(v,u), "Unit": u, "Notes": n} for m,v,u,n in rows]
summary = pd.DataFrame(pretty_rows, columns=["Metric","Value","Unit","Notes"])
summary.to_csv(OUT / "vfb_values.csv", index=False)

# pretty md/txt도 같이 저장
md = ["| Metric | Value | Unit | Notes |","|---|---:|:---:|---|"]
for r in pretty_rows:
    md.append(f"| {r['Metric']} | {r['Value']} | {r['Unit']} | {r['Notes']} |")
(OUT / "vfb_values_pretty.md").write_text("\n".join(md), encoding="utf-8")

w_metric = max(len(r["Metric"]) for r in pretty_rows)
w_value  = max(len(str(r["Value"])) for r in pretty_rows)
w_unit   = max(len(r["Unit"]) for r in pretty_rows)
w_notes  = max(len(r["Notes"]) for r in pretty_rows) if any(r["Notes"] for r in pretty_rows) else 5
hdr = f"{'Metric'.ljust(w_metric)}  {'Value'.rjust(w_value)}  {'Unit'.center(w_unit)}  Notes"
sep = f"{'-'*w_metric}  {'-'*w_value}  {'-'*w_unit}  {'-'*max(5,w_notes)}"
lines = [hdr, sep]
for r in pretty_rows:
    lines.append(f"{r['Metric'].ljust(w_metric)}  {str(r['Value']).rjust(w_value)}  {r['Unit'].center(w_unit)}  {r['Notes']}")
(OUT / "vfb_values_pretty.txt").write_text("\n".join(lines), encoding="utf-8")

# -----------------------------
# Plots
# -----------------------------
plt.figure(); plt.plot(V, Q, marker='o'); plt.grid(True)
plt.xlabel('Gate Voltage V (V)'); plt.ylabel('Sheet charge $Q_s$ (C/cm$^2$)')
plt.title('Q–V'); plt.tight_layout(); plt.savefig(OUT/"QV.png", dpi=150)

plt.figure(); plt.plot(V, C, marker='o'); plt.grid(True)
plt.xlabel('Gate Voltage V (V)'); plt.ylabel('Capacitance per area C (F/cm$^2$)')
plt.title('C–V (from Q–V)'); plt.tight_layout(); plt.savefig(OUT/"CV.png", dpi=150)

plt.figure(); plt.plot(V, df["C_over_Cox"], marker='o'); plt.grid(True)
plt.xlabel('Gate Voltage V (V)'); plt.ylabel('C / C$_{ox}$ (–)')
plt.title('Normalized C–V'); plt.tight_layout(); plt.savefig(OUT/"CV_norm.png", dpi=150)

print("Saved all results to:", OUT.resolve())
