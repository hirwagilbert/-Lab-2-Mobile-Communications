# Mobile Communications - Unit 2 Lab (Lab 2)

Python code, results and figures for the three Unit 2 mini projects.

| Folder | Mini project | Script |
|---|---|---|
| `Mini-Project-2.1/` | Free-space path loss (Friis) and BPSK/QPSK/16-QAM/64-QAM BER over AWGN | `mp2_1_friis_ber.py` |
| `Mini-Project-2.2/` | OFDM waveform generator with cyclic-prefix visualiser | `mp2_2_ofdm_cp.py` |
| `Mini-Project-2.3/` | WPAN / WLAN / WMAN range-throughput-power comparison tool | `mp2_3_wireless_compare.py` |

## Run

```bash
pip install -r requirements.txt
cd Mini-Project-2.1 && python mp2_1_friis_ber.py     # figures + printed results
cd ../Mini-Project-2.2 && python mp2_2_ofdm_cp.py
cd ../Mini-Project-2.3 && python mp2_3_wireless_compare.py
```

Python 3.9+. No SDR hardware required. Random seed is fixed (`np.random.seed(0)`), so results are reproducible.
Each folder also contains the figures and the console output (`*_output.txt`) produced by the last run.

## Notes
- 2.1 also covers the optional lab tasks: QPSK, 64-QAM, rate-1/2 convolutional code with hard-decision Viterbi decoding, log-distance model, log-normal shadowing and a throughput-vs-Shannon comparison.
- 2.3: each dataset entry is tagged as sourced (S) or engineering assumption (A) in the script; sources are listed in the report.

## Report
The written report is in `Lab2_Report.pdf`.

**Group members (University of Rwanda, Gako Campus, Computer Engineering, Year 4):**
Janvier NSHIMIYIMANA (223026679), Blandon ASHIMIRWE (223026675), Gilbert HIRWA NTWALI (223026689).
