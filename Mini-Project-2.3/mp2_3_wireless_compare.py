"""Mini Project 2.3 - WPAN vs WLAN vs WMAN trade-off tool (range, throughput, power)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# basis: S = taken from a cited source, A = engineering assumption (stated in report)
data = [
    dict(tech="BLE 5 (LE 1M)",        cls="WPAN", band="2.4 GHz ISM",      range_m=50,   rate_typ=1.0,   rate_peak=2.0,
         tx_dBm=0,  basis_range="A", basis_rate="S", basis_pwr="A"),
    dict(tech="Bluetooth Classic (Class 2)", cls="WPAN", band="2.4 GHz ISM", range_m=10,  rate_typ=2.1,   rate_peak=3.0,
         tx_dBm=4,  basis_range="S", basis_rate="A", basis_pwr="S"),
    dict(tech="Bluetooth Classic (Class 1)", cls="WPAN", band="2.4 GHz ISM", range_m=100, rate_typ=2.1,   rate_peak=3.0,
         tx_dBm=20, basis_range="S", basis_rate="A", basis_pwr="S"),
    dict(tech="Wi-Fi 5 (802.11ac)",   cls="WLAN", band="5 GHz",            range_m=35,   rate_typ=433,   rate_peak=6933,
         tx_dBm=20, basis_range="A", basis_rate="S", basis_pwr="A"),
    dict(tech="Wi-Fi 6 (802.11ax)",   cls="WLAN", band="2.4 / 5 GHz",      range_m=35,   rate_typ=600,   rate_peak=9608,
         tx_dBm=20, basis_range="S", basis_rate="S", basis_pwr="A"),
    dict(tech="Mobile WiMAX (802.16e)", cls="WMAN", band="2.5 / 3.5 GHz",  range_m=3000, rate_typ=4.0,   rate_peak=30,
         tx_dBm=23, basis_range="A", basis_rate="S", basis_pwr="A"),
]
df = pd.DataFrame(data)
df["tx_mW"] = 10**(df.tx_dBm/10)
df["rate_x_range"] = df.rate_typ * df.range_m              # Mbps*m figure of merit
df["mW_per_Mbps"] = df.tx_mW / df.rate_typ                 # transmit-power cost of a Mbps
df["spectral_note"] = df.apply(lambda r: f"{r.rate_typ:g} Mbps @ {r.range_m:g} m", axis=1)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(df[["tech", "cls", "band", "range_m", "rate_typ", "rate_peak", "tx_dBm", "tx_mW",
          "rate_x_range", "mW_per_Mbps"]].round(2).to_string(index=False))
df.to_csv("wireless_dataset.csv", index=False)

# ---------------- chart 1: range vs throughput, marker size = Tx power ----------------
colors = {"WPAN": "tab:green", "WLAN": "tab:blue", "WMAN": "tab:red"}
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for _, r in df.iterrows():
    ax.scatter(r.range_m, r.rate_typ, s=60 + 25*r.tx_dBm**1.3 if r.tx_dBm > 0 else 60,
               color=colors[r.cls], alpha=.6, edgecolor="k")
    off = (-95, -48) if r.tech.startswith("Wi-Fi 5") else (28, 6) if r.tech.startswith("Wi-Fi 6") else (8, 6)
    ax.annotate(f"{r.tech}\n{r.tx_dBm:g} dBm", (r.range_m, r.rate_typ), xytext=off,
                textcoords="offset points", fontsize=7.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(3, 2e4); ax.set_ylim(0.4, 2500)
ax.set_xlabel("Typical range (m, log scale)"); ax.set_ylabel("Typical throughput (Mbps, log scale)")
ax.grid(True, which="both", alpha=.3)
for k, c in colors.items():
    ax.scatter([], [], color=c, label=k, edgecolor="k")
ax.legend(title="Network class  (marker size ~ Tx power)", loc="upper right")
ax.set_title("Range - throughput - power trade-off space")
fig.tight_layout(); fig.savefig("fig_2_3_tradeoff.png", dpi=200); plt.close(fig)

# ---------------- chart 2: power cost per Mbps ----------------
fig, ax = plt.subplots(figsize=(8.5, 4.2))
o = df.sort_values("mW_per_Mbps")
ax.barh(o.tech, o.mW_per_Mbps, color=[colors[c] for c in o.cls])
ax.set_xscale("log"); ax.set_xlabel("Transmit power per Mbps of typical throughput (mW/Mbps, log)")
ax.grid(True, axis="x", which="both", alpha=.3); ax.set_title("Transmit-power cost of throughput")
fig.tight_layout(); fig.savefig("fig_2_3_power.png", dpi=200); plt.close(fig)

# ---------------- scenarios ----------------
scen = [
    dict(name="S1 Smart home sensors", range_m=20,   rate=0.1, max_tx_dBm=4,  note="30 coin-cell sensors, one hub, <1 kbps-100 kbps each"),
    dict(name="S2 Campus network",     range_m=30,   rate=100, max_tx_dBm=30, note="Mains-powered APs, >=100 Mbps/user, hundreds of users"),
    dict(name="S3 Metro fixed link",   range_m=3000, rate=4,   max_tx_dBm=43, note="Sites ~3 km from a base station, ~4 Mbps each"),
]
print("\n=== Scenario feasibility (range, typical rate, Tx-power limit all satisfied) ===")
rows = []
for s in scen:
    ok = df[(df.range_m >= s["range_m"]) & (df.rate_typ >= s["rate"]) & (df.tx_dBm <= s["max_tx_dBm"])].copy()
    ok = ok.sort_values("mW_per_Mbps" if s["name"].startswith("S1") else "rate_typ",
                        ascending=s["name"].startswith("S1"))
    print(f"\n{s['name']}: need >= {s['range_m']} m, >= {s['rate']} Mbps, Tx <= {s['max_tx_dBm']} dBm")
    print("  feasible:", ", ".join(ok.tech) if len(ok) else "none")
    for _, r in df.iterrows():
        fails = []
        if r.range_m < s["range_m"]: fails.append("range")
        if r.rate_typ < s["rate"]: fails.append("rate")
        if r.tx_dBm > s["max_tx_dBm"]: fails.append("power")
        rows.append(dict(scenario=s["name"], tech=r.tech, fails=",".join(fails) or "-"))
pd.DataFrame(rows).to_csv("scenario_matrix.csv", index=False)
