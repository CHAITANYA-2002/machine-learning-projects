# Gait-Trajectory Sequence-Modeling Study

> **Project status — source data required.** This repository preserves three exploratory notebooks and a visual asset, but none of the CSV files they reference is available locally. No trained-model, metric, or clinical-capability claim is made here.

## The first question is not “which neural network?”

Gait data can be clinically, personally, and contextually sensitive. Before choosing a DNN or LSTM, an experiment must establish what a row or sequence means, which sensor or motion-capture procedure produced it, what the target represents, whether participants consented to reuse, and how participants are kept separate between training and evaluation. Without those facts, a low loss is uninterpretable.

This is a research/prototyping archive. It is **not** a diagnostic, fall-risk, rehabilitation, identity, or treatment recommendation system.

## Evidence ledger

| Asset | What it contains | What can be verified now |
|---|---|---|
| `DNN.ipynb` | Dense-network experiments, CSV transforms, plots | Code and missing file references inspected |
| `lstm.ipynb` | Recurrent-sequence experimentation | Code and missing file references inspected |
| `final.ipynb` | Consolidated exploratory visualizations and sequence attempts | Preserved; contains historical notebook outputs/errors |
| `gtp.png` | Project visual asset | Preserved |
| `main.py` | Unrelated sports-session scheduling utility | Preserved; not used by the gait notebooks |
| `LEGACY_README.md` | Original description | Preserved as historical context |

The notebooks reference local CSV names including `Book14.csv`, `Book13.csv`, `Book.csv`, `s1i-5.csv`, and `lstm1 ankle.csv`. They are not in this checkout, so no execution or metric claim has been made during the revamp.

## Intended experimental shape

```mermaid
flowchart LR
    A[Consented motion/sensor captures] --> B[Metadata + data dictionary]
    B --> C[Quality checks and segmentation]
    C --> D[Participant-level train/validation/test split]
    D --> E[Training-only scaling and windowing]
    E --> F{Baseline family}
    F --> G[Dense model for fixed features]
    F --> H[LSTM for ordered windows]
    G --> I[Held-out sequence evaluation]
    H --> I
    I --> J[Error + uncertainty analysis]
```

The comparison is meaningful only after a baseline and the split are fixed. A dense neural network can model fixed-length engineered features; an LSTM can model ordered samples and temporal dependence. Neither is automatically better—an LSTM is inappropriate if rows have no consistent time order or if windows mix participants.

## What the archived notebooks attempt

The DNN notebook loads CSV data, selects numeric columns, and constructs a multi-output dense network with ReLU hidden layers and a linear output layer. It uses mean-squared error as a **loss**, not as an activation function (the legacy README conflated those concepts). It also writes intermediate CSV files such as `train.csv`, `test.csv`, and `data.csv` without recording their input schema or provenance.

The LSTM work indicates an intention to learn trajectories from ordered samples. For that intention to be valid, every sample must specify at least a participant identifier, trial/session, timestamp or frame order, coordinate convention, sampling rate, and target units. A sequence model must be split by participant and trial before overlapping windows are generated; otherwise adjacent windows leak almost identical motion into evaluation.

```text
one participant / one trial
  raw frames ──► quality check ──► ordered window ──► target at horizon h
                                  │
                                  └──── belongs to exactly one split
```

## Recovery data contract

Place authorised source data under ignored `data/`, with a small, non-sensitive data dictionary checked into the project:

```text
data/
  raw/                       # local, access-controlled source CSVs
  manifests/
    participants.csv          # pseudonymous ID, consent/scope, cohort metadata
    trials.csv                # trial ID, participant ID, device, rate, split
  processed/                  # generated locally; not committed
docs/
  data-dictionary.md          # units, axes, target definition, missing-value code
```

The manifest must assign whole participants (and preferably all closely related sessions) to train, validation, or test. It must never infer medical labels from unverified files or make sensitive attributes public.

## Evaluation plan—not an existing result

The right metric depends on the target:

| Target type | Minimum report |
|---|---|
| Future coordinate/trajectory | MAE and RMSE in stated physical units, horizon-wise error, trajectory plot with uncertainty |
| Discrete gait event/class | Per-class precision/recall/F1, confusion matrix, class counts |
| Continuous clinical score | MAE/RMSE, calibration or residual analysis, participant-level confidence intervals |

Every result should compare against simple baselines—last-value persistence, mean trajectory, and a non-sequential regression where appropriate. Aggregate metrics can hide a model that fails for particular devices, speeds, sessions, or participants, so evaluation needs those breakdowns only where privacy and sample sizes allow.

## Known gaps and regression guards

| Observed issue | Why it matters | Required guard |
|---|---|---|
| Source CSVs are absent | The code’s inputs and targets cannot be checked | Versioned data manifest and dictionary |
| Multiple ad-hoc filenames | Data lineage is ambiguous | Single config/CLI input, no notebook-local filenames |
| Notebook outputs include historical failures | Old outputs can be mistaken for current evidence | Execute cleanly from a fresh kernel and retain logs |
| No visible participant-level split contract | Sequence leakage produces misleadingly low error | Group split before normalisation/windowing |
| `main.py` is unrelated | A project entry point could confuse reviewers | Keep it labelled as a separate utility or move it in a future scoped cleanup |

## Reproduction status

Do not run the notebooks expecting a result until the source dataset is restored and a data dictionary is reviewed. The minimal historical environment is:

```bash
cd "Gait trajectory"
python -m pip install -r requirements.txt
jupyter notebook
```

This establishes packages only. It does not authorize or replace the missing dataset, participant metadata, or evaluation protocol.

## Professional conclusion

The repository preserves useful exploratory work around dense and recurrent models. The next professional milestone is not more epochs: it is a documented data contract and participant-safe evaluation design. Until then, the project remains an unverified research archive rather than a working gait predictor.

Read the [technical walkthrough](docs/index.html) for the full data-flow and recovery plan.
