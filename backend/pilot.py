import sys, os, time
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_engine import PrivacyEngine

def run_pilot():
    ulb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "creditcard.csv")
    df = pd.read_csv(ulb_path)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    
    y = df["Class"]
    X = df.drop(columns=["Class"])
    
    engine = PrivacyEngine(seed=42)
    
    start_time = time.time()
    out = engine.run_pipeline(
        X=X,
        y=y,
        epsilon=1.0,
        delta=0.00001,
        sensitivity=6.0,  # Corrected sensitivity
        mechanism="laplace_clipped",
        classifier_type="logistic_regression",
        test_size=0.20,
        seed=42,
        n_shadow=5,
    )
    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    run_pilot()
