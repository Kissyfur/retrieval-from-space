# Retrieval From Space

Config-driven tooling for building target-agnostic Copernicus matchup datasets and training models from them.

The project is no longer centered on one target such as Pseudo-nitzschia or phosphate. A user supplies a target table, Copernicus product definitions, preprocessing options, and the problem type. The tool then saves every stage in a reproducible run folder.

## Run Layout

Each run is saved under `outputs/runs/<run_id>/`:

```text
config/config.json
logs/
raw/
processed/targets.csv
processed/matchups/
datasets/
models/
metrics/
reports/
checkpoints/
pipeline_state.json
```

## Main Commands

```bash
python bin/download_copernicus.py --config configs/example_regression.yaml
python bin/create_matchups.py --config configs/example_regression.yaml --run-id <run_id>
python bin/preprocess_dataset.py --config configs/example_regression.yaml --run-id <run_id>
python bin/train_model.py --config configs/example_regression.yaml --run-id <run_id>
python bin/evaluate_model.py --config configs/example_regression.yaml --run-id <run_id>
```

Or run the whole chain:

```bash
python bin/run_pipeline.py --config configs/example_regression.yaml
```

## Synthetic End-To-End Smoke Test

Create synthetic target observations plus local Copernicus-like NetCDF products, then run every stage:

```powershell
.\.venv\Scripts\Activate.ps1
python bin/run_synthetic_example.py
```

This writes a named/versioned run such as:

```text
outputs/runs/<timestamp>_synthetic_end_to_end_v1/
```

The run includes `run_manifest.json`, `config/config.json`, `pipeline_state.json`, raw products, matchups, datasets, model artifacts, metrics, predictions, and a report.

If `problem.type` is omitted, training can ask interactively:

```bash
python bin/train_model.py --config configs/my_config.yaml --run-id <run_id> --ask-problem-type
```

## Target Table

The target table can be CSV or Excel. Configure the columns that represent:

- id
- latitude
- longitude
- time
- target value or class

The loader standardizes them internally to `Id`, `lat`, `lon`, `time`, and `target`.

## Configuration

See:

- `configs/example_regression.yaml`
- `configs/example_classification.yaml`
- `configs/synthetic_end_to_end.yaml`
- `configs/products/mediterranean_products.yaml`

Products are configured by `dataset_id` or fallback `dataset_ids`, variables, feature group, and preprocessing options. Product outputs become NetCDF files in the run folder.

## Training Strategies

The default strategy is `direct`: one model uses the configured feature groups and predicts the target.

```yaml
model:
  strategy: direct
  family: random_forest
  feature_groups: [optics, phy, meta]
```

For late fusion, use `stacking`. The base model is trained on Copernicus feature groups. Out-of-fold base predictions are then combined with metadata features to train the final model, avoiding leakage from fitting and predicting on the same rows.

```yaml
model:
  strategy: stacking
  include_base_prediction: true
  base_model:
    family: random_forest
    feature_groups: [optics, phy, nut]
  final_model:
    family: random_forest
    feature_groups: [meta]
```

For regression, `residual_correction` trains the final model to predict `target - base_prediction`, then adds that correction back to the base prediction.

```yaml
model:
  strategy: residual_correction
  base_model:
    family: random_forest
    feature_groups: [optics, phy]
  final_model:
    family: random_forest
    feature_groups: [meta]
```

Each model stage can search a candidate pool with cross-validation. Scoring uses scikit-learn scorer names, where higher is better.

```yaml
hyperparameter_search:
  enabled: true
  cv: 5
  scoring: r2
  candidates:
    - n_estimators: 200
      max_depth: 8
    - n_estimators: 400
      max_depth:
```

## Notes

Raw local data and generated artifacts are intentionally ignored by git. Notebooks were removed from the active project surface; reusable logic now lives under `src/retrieval_from_space/`, and user-facing commands live under `bin/`.

## GPU Setup

For native Windows TensorFlow GPU, keep the old TensorFlow/Keras pins in `requirements.txt`. For the modern official TensorFlow GPU route on a Windows machine, use WSL2 instead. Details are in `docs/gpu_setup.md`.
