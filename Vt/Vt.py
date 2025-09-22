# -*- coding: utf-8 -*-
"""
Vt_from_linear_ns.py
- 입력 텍스트에서 x=sweep_index(게이트 전압 V), y=Cl_3[1/cm^2](전자 시트농도 n_s)만 사용
- 아이디어: 강한 반전 구간(고 V)에서 n_s ~ (Cox/q)*(V - Vt) => V ~ (q/Cox)*n_s + Vt
  -> V 대 n_s를 직선으로 피팅하면 y-절편이 Vt, 기울기 = dV/dn_s ≈ q/Cox
- 자동으로 고V 직선 구간을 잡고(RANSAC 유사 방식), 선형 피팅 + 플롯 저장
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 파일 경로 (필요시 수정)
# -----------------------------
INFILE = Path("first_crossing_x.dat")
OUT_DIR = Path("out_vt_linear")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 데이터 로드 & 전처리
# -----------------------------
df = pd.read_csv(INFILE, sep=r"\s+", engine="python", comment="#", header=0)

# 컬럼 정규화: sweep_index -> V, Cl_3[...] -> ns
rename = {}
for c in df.columns:
    s = str(c).strip()
    if s.lower().startswith("sweep"): rename[c] = "V"
    if "Cl_3" in s or "Cl3" in s:     rename[c] = "ns"
df = df.rename(columns=rename)

for col in ["V","ns"]:
    if col not in df.columns:
        raise ValueError(f"필요 컬럼 누락: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["V","ns"]).sort_values("V").reset_index(drop=True)

V  = df["V"].to_numpy()    # [V]
ns = df["ns"].to_numpy()   # [cm^-2]

# -----------------------------
# 고V(큰 V) 구간 후보 선택 + outlier 억제
#   1) 상위 p% 전압 구간에서 초기 선형 피팅
#   2) 잔차 기반 인라이어만 골라 재피팅 (2σ 이내)
# -----------------------------
p_high = 0.35  # 상위 35% V 구간을 초기 후보로 (필요시 0.2~0.5 조절)
n = len(V)
cut = max(10, int(np.ceil((1.0 - p_high) * n)))
V_high  = V[cut:]
ns_high = ns[cut:]

# 초기 피팅: V = a * ns + b
a0, b0 = np.polyfit(ns_high, V_high, deg=1)  # slope=a0=dV/dns, intercept=b0≈Vt
V_pred0 = a0 * ns_high + b0
res0 = V_high - V_pred0
sigma = np.std(res0) if len(res0) > 1 else 0.0

# 인라이어 선택(잔차 2σ 이내)
if sigma > 0:
    mask_in = np.abs(res0) <= 2.0 * sigma
else:
    mask_in = np.ones_like(V_high, dtype=bool)

ns_in = ns_high[mask_in]
V_in  = V_high[mask_in]

# 재피팅(최종): V = a * ns + b
if len(ns_in) >= 2:
    a, b = np.polyfit(ns_in, V_in, deg=1)
else:
    # 데이터가 부족하면 초기값 사용
    a, b = a0, b0

Vt = b                     # 절편 = Vt
dV_dns = a                 # slope = dV/dn_s ≈ q/Cox
Cox_over_q = 1.0 / dV_dns  # (Cox/q) 추정치 (유효)

# 각 점에서의 즉시 추정치: V - slope*ns
Vt_pointwise = V - dV_dns * ns
Vt_point_avg = np.median(Vt_pointwise[cut:])  # 고V 영역 중앙값

# -----------------------------
# 로그 출력
# -----------------------------
print("==== Linear method (strong inversion) ====")
print(f"Fit model:  V ≈ a*ns + b  (a=dV/dn_s, b=Vt)")
print(f"a = dV/dn_s = {dV_dns:.6e}  [V·cm^2]")
print(f"b = Vt      = {Vt:.6f}      [V]")
print(f"(Cox/q) ≈ 1/a = {Cox_over_q:.6e}  [cm^-2/V]")
print(f"Vt (pointwise median on high-V) ≈ {Vt_point_avg:.6f} V")
print(f"Using Vt = intercept = {Vt:.6f} V")

# -----------------------------
# 플롯 1: 원래 좌표 (ns vs V) + 직선의 역변환
#   ns_fit(V) = (Cox/q)*(V - Vt) = (1/a)*(V - b)
# -----------------------------
ns_fit = Cox_over_q * (V - Vt)

plt.figure(figsize=(7.6,5.0))
plt.plot(V, ns, marker='o', linewidth=1.0, label=r'$n_s(V)$ data')
plt.plot(V, ns_fit, linestyle='--', label=r'linear fit: $n_s \approx (1/a)\,(V - V_t)$')
# 중요 지점: (Vt, 0)
plt.scatter([Vt],[0.0], s=60, zorder=5)
plt.annotate(f"Vt\n({Vt:.3f}, 0)",
             (Vt, 0.0), textcoords="offset points", xytext=(10,10))
# 샘플로 한 점 표시: 고V 중간점
j = cut + (len(V)-cut)//2
plt.scatter([V[j]],[ns[j]], s=40, zorder=5)
plt.annotate(f"sample\n({V[j]:.3f}, {ns[j]:.3e})",
             (V[j], ns[j]), textcoords="offset points", xytext=(8,10))
plt.grid(True)
plt.xlabel('Gate Voltage V (V)')
plt.ylabel(r'Electron sheet density $n_s$ (cm$^{-2}$)')
plt.title('Linear Vt from $n_s$ vs $V$ (strong inversion)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"ns_vs_V_linear_fit.png", dpi=150)

# -----------------------------
# 플롯 2: V vs ns (피팅 직선과 절편=Vt 직접 표시)
# -----------------------------
ns_line = np.linspace(ns.min(), ns.max(), 200)
V_line  = dV_dns * ns_line + Vt

plt.figure(figsize=(7.0,4.8))
plt.plot(ns, V, 'o', ms=3, label='data')
plt.plot(ns_line, V_line, '--', label=f'fit: V = a ns + Vt\nVt={Vt:.3f} V, a={dV_dns:.3e} V·cm²')
# 절편 표시: ns=0에서 (0, Vt)
plt.scatter([0.0],[Vt], s=60, zorder=5)
plt.annotate(f"intercept=Vt\n(0, {Vt:.3f})",
             (0.0, Vt), textcoords="offset points", xytext=(10,10))
plt.grid(True)
plt.xlabel(r'$n_s$ (cm$^{-2}$)')
plt.ylabel('V (V)')
plt.title('V vs $n_s$ with linear fit (intercept = $V_t$)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"V_vs_ns_linear_fit.png", dpi=150)

print("Saved figures to:", OUT_DIR.resolve())
