"""
Rebuild everything in one command:  python run_pipeline.py

1. generate_data -> data/employees.csv
2. eda           -> images/attrition_by_factor.png, reports/eda_attrition_rates.csv
3. train         -> models/attrition_model.joblib, reports/*, images/model_evaluation.png
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

for step in ["generate_data", "eda", "train"]:
    print(f"\n=== {step} ===")
    runpy.run_path(str(SRC / f"{step}.py"), run_name="__main__")

print("\nDone ✅  Launch the app with:  streamlit run app/app.py")
