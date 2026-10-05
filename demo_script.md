# RadQueue AI - Demo Script

**Duration:** 5-7 minutes. Start the dashboard first: `streamlit run dashboard/app.py`. Numbers below come from `reports/final_report.md` (Tier 1, 5 days × 10 seeded replications); rerun to refresh.

## Slide 1: Problem Statement (30 sec)
"Indian radiology departments serve 150-250 patients a day, mostly walk-ins, on scarce MRI/CT capacity. Patients wait blind, and FCFS queues let emergencies wait behind routine scans."

## Slide 2: Architecture (45 sec)
"Prediction (LightGBM/XGBoost ensemble + SHAP) feeds a three-level scheduler (day-ahead MILP, real-time dispatch, event rescheduling). Both are validated in a SimPy digital twin of the department."

## Slide 3: Live Demo — Command Center (60 sec)
*Open Command Center, drag the replay slider through the morning peak.*
"This is a simulated Tier-1 day under RadQueue AI: live queues per modality, utilisation gauges and patient flow. Alerts flag MRI saturation and any emergency that waited over 10 minutes."

## Slide 4: Live Demo — Predict a Patient's Wait (60 sec)
*Wait Time Predictor: MRI, walk-in, queue of 6.*
"The model predicts the wait with an honest 90% interval and explains why in plain language. Held-out test error is about 6.6 minutes, against 7-12 in the published literature."

## Slide 5: Live Demo — Smart Scheduler (45 sec)
*Register three routine MRI patients, then an emergency.*
"The emergency preempts the routine scan. The displaced patient keeps their place at the front of the queue, and every waiting patient gets an ETA."

## Slide 6: What-If (60 sec)
*What-If Simulator → preset "Add 1 MRI machine".*
"Same patients, one extra MRI: average wait drops about 44% and P90 about 30%. This is the evidence a hospital needs before buying equipment."

## Slide 7: Policy comparison & honesty (45 sec)
*Policy comparison tab.*
"On identical patient streams, RadQueue AI cuts emergency waits from about 34 minutes under FCFS to about 5 minutes. Average wait is unchanged and routine starvation is similar (11% vs 9% over 90 min), while SJF has the lowest average and P90 wait. Combining duration-awareness with fairness is our next scheduler improvement."

## Slide 8: Indian Context & Future Scope (30 sec)
"Tiers, holidays, monsoon and Monday surges are all configurable. Next steps: real RIS/PACS data, WhatsApp ETAs, and GPU-tuned models."
