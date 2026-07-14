======================================================================
RUN: chick_heart
======================================================================
Loading config: chick_heart
  [override] n_seeds=1
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0, n_patients=46, epochs=50

Output: outputs/benchmark/chick_heart/2026-07-14_09-47-54

--- Loading chick_heart dataset (Empirical Pipeline) ---
  signal: (23, 861, 1), null: (23, 861, 1)

System: Chick_heart Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                   181.4      0.214        nan
RunningVar                119.2      0.293        nan
Lag2-CSD                  223.2      0.964     0.0689
Kalman-BCE                232.5      0.780     0.6540
Kalman-LSTM               212.0      0.538     0.4901
Kalman-LSTM-Spec          217.0      0.617     0.6249
Kalman-LSTM-Aux           232.5      0.489     0.9035

Time: 3432.0s

System Verdicts:
   Chick_heart: FAIL

VERDICT: NO-GO (0/1)

All results saved to: outputs/benchmark/chick_heart/2026-07-14_09-47-54

======================================================================
All runs complete. Total time: 3432.1s
==

======================================================================
RUN: chick_heart
======================================================================
Loading config: chick_heart
  [override] n_seeds=1
  [override] methods=Kalman-Lag2,Kalman-Lag2-Net,Lag2-CSD-detrended
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0, n_patients=46, epochs=50

Output: outputs/benchmark/chick_heart/2026-07-14_12-54-52

--- Loading chick_heart dataset (Empirical Pipeline) ---
Downloading chick heart CSV from https://raw.githubusercontent.com/ThomasMBury/dl_discrete_bifurcation/main/data/df_chick.csv ...
Saved to /content/csd_observer/data/chick_heart/df_chick.csv
  signal: (23, 861, 1), null: (23, 861, 1)

System: Chick_heart Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended        175.3      0.966     0.0047
Kalman-Lag2               100.6      0.529     0.3200
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-LSTM-Aux             nan        nan        nan
Kalman-Lag2-Net           199.1      0.669     0.6286

Time: 164.4s

System Verdicts:
   Chick_heart: FAIL

VERDICT: NO-GO (0/1)

All results saved to: outputs/benchmark/chick_heart/2026-07-14_12-54-52

======================================================================
All runs complete. Total time: 164.6s



======================================================================
RUN: chick_heart
======================================================================
Loading config: chick_heart
  [override] n_seeds=1
  [override] methods=Kalman-ACKO
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0, n_patients=46, epochs=50

Output: outputs/benchmark/chick_heart/2026-07-14_14-37-15

--- Loading chick_heart dataset (Empirical Pipeline) ---
Downloading chick heart CSV from https://raw.githubusercontent.com/ThomasMBury/dl_discrete_bifurcation/main/data/df_chick.csv ...
Saved to /content/csd_observer/data/chick_heart/df_chick.csv
  signal: (23, 861, 1), null: (23, 861, 1)

System: Chick_heart Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-LSTM-Aux             nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO               232.4      0.678     0.6778

Time: 1215.6s

System Verdicts:
   Chick_heart: FAIL

VERDICT: NO-GO (0/1)

All results saved to: outputs/benchmark/chick_heart/2026-07-14_14-37-15

======================================================================
All runs complete. Total time: 1215.8s