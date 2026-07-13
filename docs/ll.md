
[1/3] ======================================================================
[1/3] RUN: default
[1/3] ======================================================================
Loading config: default
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=500, epochs=30

Output: outputs/benchmark/default/2026-07-13_12-56-27

--- Generating fold data ---
  signal: (500, 200, 1), null: (500, 200, 1)
                                                           
System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    71.4      0.034        nan
Kalman-BCE                 89.1      0.499     0.3080
Kalman-LSTM                57.3      0.756     0.0788
Kalman-LSTM-Spec           57.1      0.744     0.0759
--- Generating hopf data ---
  signal: (500, 200, 2), null: (500, 200, 2)
                                                           
System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    24.6      0.574        nan
Kalman-BCE                100.0      0.107     0.6198
Kalman-LSTM                76.3      0.475     0.3582
Kalman-LSTM-Spec           89.4      0.333     0.4084
--- Generating logistic data ---
  signal: (500, 200, 1), null: (500, 200, 1)
                                                               
System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    25.1      0.172        nan
Kalman-BCE                 63.2      0.242     0.2511
Kalman-LSTM                53.1      0.491     0.2518
Kalman-LSTM-Spec           53.9      0.496     0.3915

Time: 2136.2s

System Verdicts:
          Fold: PASS
          Hopf: PASS
      Logistic: PASS

VERDICT: GO (3/3)

All results saved to: outputs/benchmark/default/2026-07-13_12-56-27

[2/3] ======================================================================
[2/3] RUN: high_noise
[2/3] ======================================================================
Loading config: high_noise
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.3, n_patients=500, epochs=30

Output: outputs/benchmark/high_noise/2026-07-13_13-32-03

--- Generating fold data ---
  signal: (500, 200, 1), null: (500, 200, 1)
                                                           
System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    92.0      0.043        nan
Kalman-BCE                126.0      0.135     0.6108
Kalman-LSTM                63.7      0.312     0.3817
Kalman-LSTM-Spec           63.5      0.372     0.3245
--- Generating hopf data ---
  signal: (500, 200, 2), null: (500, 200, 2)
                                                           
System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    29.8      0.482        nan
Kalman-BCE                100.0      0.075     0.6581
Kalman-LSTM                80.9      0.316     0.4227
Kalman-LSTM-Spec           90.0      0.257     0.5996
--- Generating logistic data ---
  signal: (500, 200, 1), null: (500, 200, 1)
                                                               
System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    31.7      0.209        nan
Kalman-BCE                 66.5      0.203     0.2554
Kalman-LSTM                54.5      0.546     0.3971
Kalman-LSTM-Spec           48.7      0.550     0.2246

Time: 2108.6s

System Verdicts:
          Fold: PASS
          Hopf: PASS
      Logistic: PASS

VERDICT: GO (3/3)

All results saved to: outputs/benchmark/high_noise/2026-07-13_13-32-03

[3/3] ======================================================================
[3/3] RUN: low_data
[3/3] ======================================================================
Loading config: low_data
Device: cuda
Torch version: 2.11.0+cu128
Settings: noise_scale=0.15, n_patients=200, epochs=50

Output: outputs/benchmark/low_data/2026-07-13_14-07-12

--- Generating fold data ---
  signal: (200, 200, 1), null: (200, 200, 1)
                                                           
System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    88.6      0.075        nan
Kalman-BCE                 92.2      0.340     0.3010
Kalman-LSTM                59.0      0.614     0.0706
Kalman-LSTM-Spec           59.1      0.612     0.0701
--- Generating hopf data ---
  signal: (200, 200, 2), null: (200, 200, 2)
                                                           
System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    22.8      0.481        nan
Kalman-BCE                100.0      0.108     0.6838
Kalman-LSTM                92.7      0.253     0.5479
Kalman-LSTM-Spec           99.6      0.118     0.7287
--- Generating logistic data ---
  signal: (200, 200, 1), null: (200, 200, 1)
                                                               
System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    36.7      0.204        nan
Kalman-BCE                 66.7      0.048     0.3602
Kalman-LSTM                59.0      0.512     0.4196
Kalman-LSTM-Spec           59.0      0.538     0.4214

Time: 1403.4s

System Verdicts:
          Fold: PASS
          Hopf: PASS
      Logistic: PASS

VERDICT: GO (3/3)

All results saved to: outputs/benchmark/low_data/2026-07-13_14-07-12

======================================================================
All runs complete. Total time: 5648.4s