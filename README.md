
# Spot-The-Scam ML Backend (Ultimate V4)

## Setup Instructions

1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

2️⃣ Prepare data files in `data/` folder:
- `train.csv` (contains 'fraudulent' column)
- `test.csv` (no label needed)
- `frontend.csv` (optional for frontend testing)

3️⃣ Run full pipeline:

```bash
python run_all.py
```

This will train model, predict on test & frontend data.
