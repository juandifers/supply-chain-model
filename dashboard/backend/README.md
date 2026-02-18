# SupplySim Dashboard Backend

## Run

```bash
cd dashboard/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Set a custom scenario root if needed:

```bash
export SUPPLYSIM_SCENARIO_ROOT=/absolute/path/to/artifacts/scenarios
```

## Test

```bash
cd /path/to/repo
pytest dashboard/backend/tests -q
```
