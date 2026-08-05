
======================================================================
RUN: patients_100
======================================================================
Loading config: patients_100
  [override] n_seeds=4
  [override] methods=Kalman-LSTM-Spec,Kalman-Spectral-Drift
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=100, epochs=50

Output: outputs/benchmark/patients_100/2026-08-05_00-33-29

--- Generating fold data (Synthetic Pipeline) ---
  signal: (100, 200, 1), null: (100, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-BCE-Spec             nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec          129.1      0.500     0.5404
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan
Kalman-LSTM-Aug             nan        nan        nan
Kalman-Spectral-Drift      132.3      0.992     0.0498
--- Generating hopf data (Synthetic Pipeline) ---
  signal: (100, 200, 2), null: (100, 200, 2)

System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-BCE-Spec             nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec          100.0      0.670     0.0013
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan
Kalman-LSTM-Aug             nan        nan        nan
Kalman-Spectral-Drift       99.0      1.000     0.0485
--- Generating logistic data (Synthetic Pipeline) ---
  signal: (100, 200, 1), null: (100, 200, 1)

System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-BCE-Spec             nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec           66.7      1.000     0.1085
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan
Kalman-LSTM-Aug             nan        nan        nan
Kalman-Spectral-Drift       54.5      1.000     0.0478

Time: 340.1s

System Verdicts:
          Fold: FAIL
          Hopf: FAIL
      Logistic: FAIL

VERDICT: NO-GO (0/3)

All results saved to: outputs/benchmark/patients_100/2026-08-05_00-33-29

======================================================================
All runs complete. Total time: 340.3s