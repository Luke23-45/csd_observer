
======================================================================
RUN: default
======================================================================
Loading config: default
  [override] n_seeds=1
  [override] methods=Kalman-LSTM-Spec,Kalman-Lag2,Kalman-Lag2-Net
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=500, epochs=30

Output: outputs/benchmark/default/2026-07-15_04-12-28

--- Generating fold data (Synthetic Pipeline) ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                94.5      0.887     0.3024
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec           89.2      0.789     0.5931
Kalman-Lag2-Net           103.0      0.221     0.6240
Kalman-ACKO                 nan        nan        nan
--- Generating hopf data (Synthetic Pipeline) ---
  signal: (500, 200, 2), null: (500, 200, 2)

System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                66.6      0.591     0.8115
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec           33.5      0.992     0.0928
Kalman-Lag2-Net           100.0      0.431     0.8392
Kalman-ACKO                 nan        nan        nan
--- Generating logistic data (Synthetic Pipeline) ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                66.7      0.460     0.8301
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan      1.000     0.0000
Kalman-Lag2-Net            66.7      0.311     0.8904
Kalman-ACKO                 nan        nan        nan

Time: 244.8s

System Verdicts:
          Fold: FAIL
          Hopf: FAIL
      Logistic: FAIL

VERDICT: NO-GO (0/3)

All results saved to: outputs/benchmark/default/2026-07-15_04-12-28

======================================================================
All runs complete. Total time: 244.9s