"""Mini Project 2.1 - Free-space path loss (Friis) and BER of BPSK/QPSK/16-QAM/64-QAM over AWGN."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(0)

# ---------------- Part (a): Friis ----------------
def friis_rx_power(Ptx_dBm, Gtx_dBi, Grx_dBi, d_km, f_MHz):
    return Ptx_dBm + Gtx_dBi + Grx_dBi - 20*np.log10(d_km) - 20*np.log10(f_MHz) - 32.44

d = np.linspace(0.05, 10, 500)
freqs = [900, 1900, 3500]
plt.figure(figsize=(7, 4.5))
for f in freqs:
    plt.plot(d, friis_rx_power(43, 15, 0, d, f), label=f"{f} MHz")
plt.xscale("log"); plt.grid(True, which="both", alpha=.3)
plt.xlabel("Distance (km)"); plt.ylabel("Received power (dBm)")
plt.title("Friis received power vs distance (PTX=43 dBm, GTX=15 dBi, GRX=0 dBi)")
plt.legend(); plt.tight_layout(); plt.savefig("fig_2_1_friis.png", dpi=200); plt.close()

print("=== 6 dB-per-doubling check ===")
d1, d2 = 1.0, 2.0
for f in freqs:
    diff = friis_rx_power(43, 15, 0, d1, f) - friis_rx_power(43, 15, 0, d2, f)
    print(f"f={f} MHz: P(1 km) - P(2 km) = {diff:.2f} dB")
for f1, f2 in [(900, 1800), (1900, 3800)]:
    diff = friis_rx_power(43, 15, 0, 1.0, f1) - friis_rx_power(43, 15, 0, 1.0, f2)
    print(f"d=1 km: P({f1} MHz) - P({f2} MHz) = {diff:.2f} dB")

# ---------------- Part (b): BER ----------------
N = 399_996  # multiple of 12 so every scheme divides evenly

def gray_levels(k):
    """PAM levels for k bits, Gray-coded: returns array indexed by integer value of bit group."""
    M = 2**k
    lv = np.arange(-(M-1), M, 2)
    gray = [i ^ (i >> 1) for i in range(M)]
    table = np.zeros(M)
    for i, g in enumerate(gray):
        table[g] = lv[i]
    return table

def mod(bits, name):
    if name == "BPSK":
        return 2.0*bits - 1, 1
    k = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}[name]
    b = bits.reshape(-1, k)
    h = k // 2
    w = 2**np.arange(h-1, -1, -1)
    tab = gray_levels(h)
    I = tab[b[:, :h] @ w]; Q = tab[b[:, h:] @ w]
    s = I + 1j*Q
    s /= np.sqrt(np.mean(np.abs(s)**2))   # unit average symbol energy (Es=1)
    return s, k

def demod(r, name, k):
    if name == "BPSK":
        return (r.real > 0).astype(int)
    h = k // 2
    M = 2**h
    tab = gray_levels(h)
    norm = np.sqrt(2*(M**2-1)/3)          # same normalisation as mapper
    lv = tab / norm
    def dec(x):
        idx = np.argmin(np.abs(x[:, None] - lv[None, :]), axis=1)
        # integer index -> bits
        return ((idx[:, None] >> np.arange(h-1, -1, -1)) & 1)
    bi = dec(r.real); bq = dec(r.imag)
    return np.hstack([bi, bq]).reshape(-1)

def awgn(s, ebn0_db, k):
    ebn0 = 10**(ebn0_db/10)
    n0 = 1/(k*ebn0)                      # Es=1, Eb=1/k
    if np.isrealobj(s):
        return s + np.sqrt(n0/2)*np.random.randn(len(s))
    return s + np.sqrt(n0/2)*(np.random.randn(len(s)) + 1j*np.random.randn(len(s)))

bits = np.random.randint(0, 2, size=N)
ebn0_db = np.arange(0, 17, 1)
schemes = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
ber = {}
for name in schemes:
    s, k = mod(bits, name)
    out = []
    for e in ebn0_db:
        rb = demod(awgn(s, e, k), name, k)
        out.append(np.mean(rb != bits))
    ber[name] = np.array(out)

lin = 10**(ebn0_db/10)
theory_bpsk = 0.5*erfc(np.sqrt(lin))

plt.figure(figsize=(7, 4.8))
plt.semilogy(ebn0_db, np.maximum(ber["BPSK"], 1e-7), "o-", label="BPSK (simulated)")
plt.semilogy(ebn0_db, theory_bpsk, "k--", label="BPSK theory 0.5 erfc(sqrt(Eb/N0))")
plt.semilogy(ebn0_db, np.maximum(ber["QPSK"], 1e-7), "s-", label="QPSK (simulated)")
plt.semilogy(ebn0_db, np.maximum(ber["16-QAM"], 1e-7), "^-", label="16-QAM (simulated)")
plt.semilogy(ebn0_db, np.maximum(ber["64-QAM"], 1e-7), "d-", label="64-QAM (simulated)")
plt.ylim(1e-5, 1); plt.grid(True, which="both", alpha=.3)
plt.xlabel("Eb/N0 (dB)"); plt.ylabel("Bit error rate")
plt.title("BER vs Eb/N0 over AWGN"); plt.legend(loc="lower left", fontsize=8); plt.tight_layout()
plt.savefig("fig_2_1_ber.png", dpi=200); plt.close()

def ebn0_at(b, target, x=None):
    """Interpolate Eb/N0 (dB) where BER crosses target (log-domain)."""
    x = ebn0_db if x is None else x
    ok = b > 0
    return float(np.interp(np.log10(target), np.log10(b[ok])[::-1], x[ok][::-1]))

print("\n=== BER results (N=%d bits) ===" % N)
print("Eb/N0(dB) | " + " | ".join(f"{s:>9}" for s in schemes) + " | BPSK theory")
for i, e in enumerate(ebn0_db):
    print(f"{e:9d} | " + " | ".join(f"{ber[s][i]:9.2e}" for s in schemes) + f" | {theory_bpsk[i]:.2e}")
tgt = 1e-3
req = {s: ebn0_at(ber[s], tgt) for s in schemes}
print(f"\nEb/N0 required for BER=1e-3: " + ", ".join(f"{s}={v:.2f} dB" for s, v in req.items()))
print(f"Gap 16-QAM vs BPSK at 1e-3: {req['16-QAM']-req['BPSK']:.2f} dB")
print(f"Gap 64-QAM vs BPSK at 1e-3: {req['64-QAM']-req['BPSK']:.2f} dB")
print(f"Gap 64-QAM vs 16-QAM at 1e-3: {req['64-QAM']-req['16-QAM']:.2f} dB")

# ---------------- Lab task 2: rate-1/2 conv code (K=3, 7,5) + hard Viterbi ----------------
def conv_encode(u):
    s1 = s2 = 0; out = []
    for b in u:
        out += [b ^ s1 ^ s2, b ^ s2]
        s2 = s1; s1 = b
    return np.array(out)

def viterbi(r):
    n = len(r)//2
    INF = 1e9
    pm = np.full(4, INF); pm[0] = 0
    prev = np.zeros((n, 4), int); inp = np.zeros((n, 4), int)
    for t in range(n):
        new = np.full(4, INF)
        for st in range(4):
            if pm[st] >= INF: continue
            s1, s2 = st >> 1, st & 1
            for b in (0, 1):
                o = (b ^ s1 ^ s2, b ^ s2)
                ns = (b << 1) | s1
                m = pm[st] + (o[0] != r[2*t]) + (o[1] != r[2*t+1])
                if m < new[ns]:
                    new[ns] = m; prev[t, ns] = st; inp[t, ns] = b
        pm = new
    st = int(np.argmin(pm)); out = np.zeros(n, int)
    for t in range(n-1, -1, -1):
        out[t] = inp[t, st]; st = prev[t, st]
    return out

Nc = 150_000
u = np.random.randint(0, 2, Nc)
c = conv_encode(u)
ber_c = []
for e in ebn0_db[:9]:
    rate = 0.5
    ebn0 = 10**(e/10)
    n0 = 1/(rate*ebn0)  # Es=1 per coded bit, Eb = Es/rate
    rx = (2.0*c - 1) + np.sqrt(n0/2)*np.random.randn(len(c))
    dec = viterbi((rx > 0).astype(int))
    ber_c.append(np.mean(dec != u))
ber_c = np.array(ber_c)
plt.figure(figsize=(7, 4.5))
plt.semilogy(ebn0_db[:9], np.maximum(ber["BPSK"][:9], 1e-6), "o-", label="Uncoded BPSK")
plt.semilogy(ebn0_db[:9], np.maximum(ber_c, 1e-6), "s-", label="Rate-1/2 conv. + Viterbi (hard)")
plt.grid(True, which="both", alpha=.3); plt.ylim(1e-6, 1)
plt.xlabel("Eb/N0 (dB)"); plt.ylabel("BER"); plt.legend(); plt.title("Coded vs uncoded BPSK")
plt.tight_layout(); plt.savefig("fig_2_1_coding.png", dpi=200); plt.close()
print("\nCoded BPSK BER:", {int(e): f"{b:.2e}" for e, b in zip(ebn0_db[:9], ber_c)})
xs = ebn0_db[:9]
cg = ebn0_at(ber["BPSK"][:9], 1e-3, xs) - ebn0_at(ber_c, 1e-3, xs)
print(f"Eb/N0 for BER=1e-3: uncoded BPSK {ebn0_at(ber['BPSK'][:9],1e-3,xs):.2f} dB, coded {ebn0_at(ber_c,1e-3,xs):.2f} dB")
print(f"Approx. coding gain at 1e-3: {cg:.2f} dB")
# Es/N0 (per-symbol SNR) gap, which is the quantity the manual's 8-10 dB figure refers to
print(f"Es/N0 gap 16-QAM vs BPSK at 1e-3: {req['16-QAM']+10*np.log10(4)-req['BPSK']:.2f} dB")
print(f"Es/N0 gap 64-QAM vs BPSK at 1e-3: {req['64-QAM']+10*np.log10(6)-req['BPSK']:.2f} dB")

# ---------------- Lab tasks 4-5: log-distance + shadowing ----------------
f = 900; d0 = 0.1
PL0 = -(friis_rx_power(0, 0, 0, d0, f))
plt.figure(figsize=(7, 4.5))
for n in (2, 3, 4):
    PL = PL0 + 10*n*np.log10(d/d0)
    plt.plot(d, 43+15-PL, label=f"n = {n}")
plt.xscale("log"); plt.grid(True, which="both", alpha=.3)
plt.xlabel("Distance (km)"); plt.ylabel("Received power (dBm)")
plt.title("Log-distance model, 900 MHz"); plt.legend(); plt.tight_layout()
plt.savefig("fig_2_1_logdist.png", dpi=200); plt.close()

plt.figure(figsize=(7, 4.5))
n = 3
for i in range(5):
    sh = np.random.randn(len(d))*6   # sigma = 6 dB
    plt.plot(d, 43+15-(PL0 + 10*n*np.log10(d/d0)) + sh, lw=.8, alpha=.8)
plt.plot(d, 43+15-(PL0 + 10*n*np.log10(d/d0)), "k", lw=2, label="mean (n=3)")
plt.xscale("log"); plt.grid(True, which="both", alpha=.3); plt.legend()
plt.xlabel("Distance (km)"); plt.ylabel("Received power (dBm)")
plt.title("Log-normal shadowing, sigma = 6 dB, 5 realisations")
plt.tight_layout(); plt.savefig("fig_2_1_shadowing.png", dpi=200); plt.close()

# ---------------- Lab task 6: throughput proxy vs Shannon ----------------
plt.figure(figsize=(7, 4.5))
snr_db = ebn0_db  # use Eb/N0 axis; Es/N0 = k*Eb/N0 handled per scheme
for name in schemes:
    k = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}[name]
    plt.plot(ebn0_db + 10*np.log10(k), k*(1-ber[name]), label=name)
x = np.linspace(0, 28, 200)
plt.plot(x, np.log2(1+10**(x/10)), "k--", label="Shannon C/B")
plt.xlabel("SNR per symbol Es/N0 (dB)"); plt.ylabel("bits / symbol")
plt.grid(alpha=.3); plt.legend(); plt.title("Throughput proxy vs Shannon-Hartley")
plt.tight_layout(); plt.savefig("fig_2_1_shannon.png", dpi=200); plt.close()
