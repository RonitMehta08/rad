"""What-if scenario runner.

Owner: P4
"""
import pandas as pd
from typing import Any
from src.simulation.engine import RadiologySimulation
from src.scheduler.policies import (
    FCFSPolicy, PriorityPolicy, SJFPolicy, RadQueueAIPolicy
)

def run_scenarios(output_dir: str):
    policies = {
        "FCFS": FCFSPolicy(),
        "Priority": PriorityPolicy(),
        "SJF": SJFPolicy(),
        "RadQueue AI": RadQueueAIPolicy()
    }
    
    results = []
    
    for name, policy in policies.items():
        sim = RadiologySimulation(policy=policy)
        kpi = sim.run(until=1440)
        
        # Calculate summary metrics
        df = pd.DataFrame(kpi.patient_records)
        if not df.empty and "wait_time_minutes" in df.columns:
            avg_wait = df["wait_time_minutes"].mean()
            max_wait = df["wait_time_minutes"].max()
        else:
            avg_wait = 0
            max_wait = 0
            
        results.append({
            "Policy": name,
            "Avg Wait (min)": avg_wait,
            "Max Wait (min)": max_wait,
            "Patients Served": len(df)
        })
        
    df_res = pd.DataFrame(results)
    
    # Save results
    out_path = f"{output_dir}/simulation_comparison.csv"
    import os
    os.makedirs(output_dir, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"Scenario results saved to {out_path}")
    return df_res

if __name__ == "__main__":
    run_scenarios("reports")
