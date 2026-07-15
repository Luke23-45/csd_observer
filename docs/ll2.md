
======================================================================
RUN: default
======================================================================
Loading config: default
  [override] n_seeds=1
  [override] methods=Kalman-ACKO,Kalman-LSTM,Kalman-LSTM-Spec,Kalman-Lag2-Net
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=500, epochs=30

Output: outputs/benchmark/default/2026-07-15_03-18-25

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
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                89.2      0.791     0.5930
Kalman-LSTM-Spec           96.1      0.788     0.6162
Kalman-Lag2-Net           102.0      0.519     0.6637
Kalman-ACKO               118.2      0.783     0.2046
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
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                37.2      0.963     0.1073
Kalman-LSTM-Spec            nan      0.490     0.0000
Kalman-Lag2-Net            66.6      0.723     0.8159
Kalman-ACKO                 nan      0.940     0.0000
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
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan      1.000     0.0000
Kalman-LSTM-Spec            nan      1.000     0.0000
Kalman-Lag2-Net            66.7      0.629     0.7803
Kalman-ACKO                 nan      1.000     0.0000

Time: 680.9s

System Verdicts:
          Fold: FAIL
          Hopf: FAIL
      Logistic: FAIL

VERDICT: NO-GO (0/3)

All results saved to: outputs/benchmark/default/2026-07-15_03-18-25

======================================================================
All runs complete. Total time: 681.0s


======================================================================
RUN: default
======================================================================
Loading config: default
  [override] n_seeds=1
  [override] methods=Kalman-BCE,Kalman-Lag2,Lag2-CSD-detrended
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=500, epochs=30

Output: outputs/benchmark/default/2026-07-15_03-17-03

--- Generating fold data (Synthetic Pipeline) ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                     nan        nan        nan
RunningVar                  nan        nan        nan
Lag2-CSD                    nan        nan        nan
Lag2-CSD-detrended         88.1      0.828     0.0747
Kalman-Lag2                94.5      0.887     0.3024
Kalman-BCE                 85.4      0.923     0.2334
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
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
Lag2-CSD-detrended         28.4      0.567     0.0027
Kalman-Lag2                66.6      0.591     0.8115
Kalman-BCE                 89.3      0.823     0.0068
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
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
Lag2-CSD-detrended         22.5      0.477     0.0024
Kalman-Lag2                66.7      0.460     0.7883
Kalman-BCE                  nan      1.000     0.0000
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan

Time: 166.8s

System Verdicts:
          Fold: FAIL
          Hopf: FAIL
      Logistic: FAIL

VERDICT: NO-GO (0/3)

All results saved to: outputs/benchmark/default/2026-07-15_03-17-03

======================================================================
All runs complete. Total time: 166.9s


======================================================================
RUN: default
======================================================================
Loading config: default
  [override] n_seeds=1
  [override] methods=Lag2-CSD,Raw-CSD,RunningVar
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=500, epochs=30

Output: outputs/benchmark/default/2026-07-15_03-16-24

--- Generating fold data (Synthetic Pipeline) ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    71.4      0.561        nan
RunningVar                133.3      0.792        nan
Lag2-CSD                   88.5      0.820     0.1004
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan
--- Generating hopf data (Synthetic Pipeline) ---
  signal: (500, 200, 2), null: (500, 200, 2)

System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    24.6      0.766        nan
RunningVar                 55.7      0.779        nan
Lag2-CSD                   33.8      0.709     0.0091
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan
--- Generating logistic data (Synthetic Pipeline) ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    25.1      0.420        nan
RunningVar                 36.0      0.409        nan
Lag2-CSD                   22.8      0.467     0.0100
Lag2-CSD-detrended          nan        nan        nan
Kalman-Lag2                 nan        nan        nan
Kalman-BCE                  nan        nan        nan
Kalman-LSTM                 nan        nan        nan
Kalman-LSTM-Spec            nan        nan        nan
Kalman-Lag2-Net             nan        nan        nan
Kalman-ACKO                 nan        nan        nan

Time: 72.8s

System Verdicts:
          Fold: FAIL
          Hopf: FAIL
      Logistic: FAIL

VERDICT: NO-GO (0/3)

All results saved to: outputs/benchmark/default/2026-07-15_03-16-24

======================================================================
All runs complete. Total time: 72.9s