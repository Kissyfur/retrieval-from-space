# Synthetic End-To-End Example

This example creates a small local dataset that behaves like downloaded Copernicus products, then runs the complete pipeline:

1. Generate target observations and Copernicus-like NetCDF sources.
2. Run the download stage, using `source: local` to normalize those sources into `raw/`.
3. Create matchup windows.
4. Preprocess train-ready feature groups.
5. Train a regression model.
6. Save metrics, predictions, report, state, config snapshot, and run manifest.

Run it from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python bin/run_synthetic_example.py
```

The run is saved under:

```text
outputs/runs/<timestamp>_synthetic_end_to_end_v1/
```

The generated inputs are written to `examples/synthetic/generated/`.
