# RadQueue AI - Demo Script

**Duration:** 5-7 minutes

## Slide 1: Problem Statement (30 sec)
"Indian radiology departments serve 150-250 patients daily with 2.5-3 hour average turnaround time. Traditional systems use First-Come-First-Served (FCFS), leading to high variance and long waits for critical patients. RadQueue AI predicts and optimizes this."

## Slide 2: Architecture (45 sec)
"Our system uses a hybrid closed-loop architecture. The Prediction Engine (LightGBM/XGBoost) predicts wait times, and the Scheduling Engine (RadQueue AI Policy) dynamically re-routes patients to minimize wait times based on live queuing data."

## Slide 3: Live Demo — Command Center (60 sec)
*Action: Open `1_Command_Center.py`.*
"Here is the real-time command center. You can see total patients, average wait times, and live resource utilization for X-Ray, CT, MRI, and Ultrasound. It alerts staff if emergency queues build up."

## Slide 4: Live Demo — Predict a Patient's Wait (60 sec)
*Action: Open `2_Wait_Time_Predictor.py`.*
"Let's enter a walk-in MRI patient. The model predicts a wait time of 45 minutes. The SHAP summary explains that the current MRI queue length and peak hour are the main drivers for this wait."

## Slide 5: Live Demo — What-If Scenario (60 sec)
*Action: Open `4_What_If_Simulator.py`.*
"What if we change our policy from FCFS to RadQueue AI? Let's run a simulation. As you can see, average wait time drops significantly, and the maximum wait time is capped due to our fairness constraint."

## Slide 6: Indian Context (45 sec)
*Action: Open `7_Indian_Context.py`.*
"Since we're building for India, we have specific modifiers for Tier 1 Govt Hospitals, walk-in ratios, monsoon emergencies, and local holidays like Diwali and Holi that drastically affect OPD traffic."

## Slide 7: Conclusion
"RadQueue AI moves radiology from static queuing to dynamic, predictive routing, directly reducing patient suffering and improving resource utilization."
