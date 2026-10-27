"""Mini Project 2.2 - OFDM waveform generator with a cyclic-prefix visualiser."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
N = 64                 # subcarriers
NSYM = 400             # OFDM symbols in the stream
SNR_DB = 35            # small AWGN so the "good" case is near-perfect but realistic

# ---------- constellation (Gray-coded QPSK, unit energy) ----------
def qpsk_map(bits):
    b = bits.reshape(-1, 2)
    return ((1 - 2*b[:, 0]) + 1j*(1 - 2*b[:, 1])) / np.sqrt(2)   # 00->+,+  01->+,-  11->-,-  10->-,+

def qpsk_demap(s):
    return np.stack([(s.real < 0).astype(int), (s.imag < 0).astype(int)], axis=1).reshape(-1)

# unit test of mapper
t = np.array([0, 0, 0, 1, 1, 1, 1, 0])
assert np.array_equal(qpsk_demap(qpsk_map(t)), t)
assert np.isclose(np.mean(np.abs(qpsk_map(np.random.randint(0, 2, 10000)))**2), 1, atol=0.02)
print("[OK] QPSK mapper unit test passed")

# ---------- modulator / CP ----------
def ofdm_mod(X):               # X: (nsym, N) frequency domain -> time domain
    return np.fft.ifft(X, axis=1)

def add_cp(x, L):
    return np.hstack([x[:, -L:], x]) if L > 0 else x

# noiseless self-check
Xt = qpsk_map(np.random.randint(0, 2, 2*N)).reshape(1, N)
assert np.allclose(np.fft.fft(ofdm_mod(Xt), axis=1), Xt)
print("[OK] FFT(IFFT(X)) recovers the subcarrier values exactly (max err = %.1e)"
      % np.max(np.abs(np.fft.fft(ofdm_mod(Xt), axis=1) - Xt)))
xc = add_cp(ofdm_mod(Xt), 16)
assert np.allclose(xc[0, :16], xc[0, -16:])
print("[OK] CP samples equal the tail of the OFDM symbol")

# ---------- synthetic multipath channel ----------
delays = np.array([0, 3, 6])                    # samples
gains = np.array([1.0, 0.6*np.exp(1j*0.9), 0.4*np.exp(-1j*2.1)])
gains = gains / np.linalg.norm(gains)
h = np.zeros(delays.max()+1, complex)
h[delays] = gains
L_SPREAD = len(h) - 1                            # max excess delay = 6 samples
print(f"Channel taps (delay, gain): " + ", ".join(f"({d}, {g:.3f})" for d, g in zip(delays, gains)))
print(f"Max excess delay = {L_SPREAD} samples -> CP must be >= {L_SPREAD}")
H = np.fft.fft(h, N)                             # channel frequency response (known at Rx)

# ---------- transmit / receive chain ----------
bits = np.random.randint(0, 2, NSYM*N*2)
X = qpsk_map(bits).reshape(NSYM, N)

def run(L, snr_db=SNR_DB):
    tx = add_cp(ofdm_mod(X), L)                  # (NSYM, N+L)
    stream = tx.reshape(-1)
    rx = np.convolve(stream, h)[:len(stream)]    # consecutive symbols smear into each other
    p = np.mean(np.abs(stream)**2)
    n0 = p / 10**(snr_db/10)
    rx = rx + np.sqrt(n0/2)*(np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))
    rxs = rx.reshape(NSYM, N+L)[:, L:]           # discard CP (nothing to discard when L = 0)
    Y = np.fft.fft(rxs, axis=1) / H              # one-tap zero-forcing equaliser
    Y = Y[1:]                                    # drop first symbol (start-up transient)
    Xr = X[1:]
    ser = np.mean((np.sign(Y.real) != np.sign(Xr.real)) | (np.sign(Y.imag) != np.sign(Xr.imag)))
    mse = np.mean(np.abs(Y - Xr)**2)
    return Y, ser, mse

L_good, L_bad = 16, 2
Yg, ser_g, mse_g = run(L_good)
Yb, ser_b, mse_b = run(L_bad)
Y0, ser_0, mse_0 = run(0)
print(f"\nCP=16 (sufficient): SER={ser_g:.4f}  MSE={mse_g:.2e}")
print(f"CP=2  (too short) : SER={ser_b:.4f}  MSE={mse_b:.2e}")
print(f"CP=0  (none)      : SER={ser_0:.4f}  MSE={mse_0:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
for a, Y, ttl in [(ax[0], Yg, f"CP = {L_good} (≥ delay spread {L_SPREAD})"),
                  (ax[1], Yb, f"CP = {L_bad} (< delay spread {L_SPREAD})")]:
    a.scatter(Y.real.ravel(), Y.imag.ravel(), s=2, alpha=.35)
    a.set_title(ttl); a.set_xlabel("In-phase"); a.set_ylabel("Quadrature")
    a.grid(alpha=.3); a.set_aspect("equal")
ax[0].set_xlim(-2, 2); ax[0].set_ylim(-2, 2)
fig.suptitle("Recovered constellations after equalisation, same channel realisation")
fig.tight_layout(); fig.savefig("fig_2_2_constellations.png", dpi=200); plt.close(fig)

# ---------- sweep CP length ----------
cps = np.arange(0, 17)
sers, mses = [], []
for L in cps:
    _, s, m = run(L)
    sers.append(s); mses.append(m)
print("\nCP length | SER | MSE")
for L, s, m in zip(cps, sers, mses):
    print(f"{L:9d} | {s:.4f} | {m:.3e}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogy(cps, np.maximum(sers, 1e-5), "o-")
ax[1].semilogy(cps, mses, "s-", color="tab:red")
for a, yl in zip(ax, ["Symbol error rate", "MSE of constellation points"]):
    a.axvline(L_SPREAD, color="k", ls="--", label=f"delay spread = {L_SPREAD}")
    a.set_xlabel("Cyclic prefix length (samples)"); a.set_ylabel(yl)
    a.grid(True, which="both", alpha=.3); a.legend()
fig.suptitle(f"Error vs CP length (N={N}, QPSK, SNR={SNR_DB} dB)")
fig.tight_layout(); fig.savefig("fig_2_2_cp_sweep.png", dpi=200); plt.close(fig)

# ---------- time-domain waveform with CP (visual) ----------
x1 = add_cp(ofdm_mod(X[:1]), 16)[0]
plt.figure(figsize=(8, 3.5))
plt.plot(np.abs(x1), lw=1)
plt.axvspan(0, 16, color="orange", alpha=.3, label="cyclic prefix (copy of tail)")
plt.xlabel("Sample index"); plt.ylabel("|x[n]|"); plt.legend(); plt.grid(alpha=.3)
plt.title("One OFDM symbol (N=64) with 16-sample CP"); plt.tight_layout()
plt.savefig("fig_2_2_waveform.png", dpi=200); plt.close()
