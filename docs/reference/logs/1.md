Device: cuda
Torch version: 2.11.0+cu128

================================================================================
Stress Test: Default
  noise_scale=0.15, n_patients=500, epochs=30
================================================================================

--- Generating fold data ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    71.4      0.034        nan
Kalman-BCE                 85.1      0.202     0.4648
Kalman-LSTM                28.1      0.779     0.0415
Kalman-LSTM-Spec           28.2      0.770     0.0444

--- Generating hopf data ---
  signal: (500, 200, 2), null: (500, 200, 2)

System: Hopf Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    24.6      0.574        nan
Kalman-BCE                 98.0      0.116     0.8306
Kalman-LSTM                74.9      0.180     0.5999
Kalman-LSTM-Spec           89.8      0.174     0.7820

--- Generating logistic data ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Logistic Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    25.1      0.172        nan
Kalman-BCE                 50.3      0.144     0.2289
Kalman-LSTM                28.2      0.452     0.1367
Kalman-LSTM-Spec           28.0      0.485     0.1210

Time: 2223.7s

System Verdicts:
          Fold: PASS
          Hopf: PASS
      Logistic: PASS

VERDICT: GO - LSTM head passes on 3/3 bifurcation systems.
A causal LSTM head enables temporal CSD pattern detection beyond the per-step MLP.

Details:
  Raw-CSD detection time:            71.4
  Kalman-BCE detection time:         85.1
  Kalman-LSTM detection time:        28.1
  Kalman-LSTM-Spec detection time:   28.2
  DT gain (LSTM vs BCE):             57.0
  EW-AUC gain (LSTM vs BCE):         0.577
  FPR ratio (LSTM/BCE null):         0.089
  DT PASS: gain=57.0 >= 20
  EW-AUC PASS: gain=0.577 >= 0.05
  NULL PASS: FPR ratio=0.089
  Raw-CSD detection time:            24.6
  Kalman-BCE detection time:         98.0
  Kalman-LSTM detection time:        74.9
  Kalman-LSTM-Spec detection time:   89.8
  DT gain (LSTM vs BCE):             23.1
  EW-AUC gain (LSTM vs BCE):         0.064
  FPR ratio (LSTM/BCE null):         0.722
  DT PASS: gain=23.1 >= 20
  EW-AUC PASS: gain=0.064 >= 0.05
  NULL PASS: FPR ratio=0.722
  Raw-CSD detection time:            25.1
  Kalman-BCE detection time:         50.3
  Kalman-LSTM detection time:        28.2
  Kalman-LSTM-Spec detection time:   28.0
  DT gain (LSTM vs BCE):             22.1
  EW-AUC gain (LSTM vs BCE):         0.307
  FPR ratio (LSTM/BCE null):         0.597
  DT PASS: gain=22.1 >= 20
  EW-AUC PASS: gain=0.307 >= 0.05
  NULL PASS: FPR ratio=0.597

================================================================================
Stress Test: HighNoise
  noise_scale=0.3, n_patients=500, epochs=30
================================================================================

--- Generating fold data ---
  signal: (500, 200, 1), null: (500, 200, 1)

System: Fold Bifurcation
----------------------------------------------------------------------
Method                       DT     EW-AUC        FPR
--------------------------------------------------
Raw-CSD                    92.0      0.043        nan
Kalman-BCE                122.0      0.087     0.6569
Kalman-LSTM                37.7      0.362     0.1950
Kalman-LSTM-Spec           36.2      0.359     0.2153

--- Generating hopf data ---
  signal: (500, 200, 2), null: (500, 200, 2)
