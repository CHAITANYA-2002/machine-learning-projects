# Machine Learning Portfolio Revamp Status

This document distinguishes verified local work from projects whose original
datasets, model weights, or cloud-runtime paths are not present in this repository.
It is an audit trail—not a claim that unavailable projects have been executed.

| Project | Category | Local execution state | Required recovery input |
|---|---|---|---|
| Cervical Cancer Risk-Factor Modeling | Tabular classification | Verified | Included CSV |
| California House Price Predictor | Regression | Verified | scikit-learn public dataset fetch |
| NIFTY Closing-Price Analysis | Time-series regression | Verified | Included `NSEI.csv` |
| Fake News Classifier | NLP classification | Implementation and tests verified; full training blocked | `fake_or_real_news.csv` |
| Bluetooth Sale Predictor | Tabular regression | Documentation and recovery path complete; execution blocked | Kaggle Blue Book for Bulldozers data |
| Background Remover | Image segmentation | Documentation and recovery path complete; execution blocked | COCO 2017 images and annotations; compatible model runtime/GPU |
| COVID-19 Predictor | Image classification | Documentation and recovery path complete; execution blocked | Authorised image data plus patient/site split metadata |
| Gait-Trajectory Sequence-Modeling Study | Sequence modeling | Documentation and recovery path complete; execution blocked | Authorised source CSVs, data dictionary, and participant/trial split metadata |
| Image Enhancer | Super-resolution | Documentation and recovery path complete; inference blocked | ESRGAN/RRDB pretrained weights |
| Histology Colour-Normalisation Exploration | Medical-image normalization | Documentation and recovery path complete; execution blocked | Original Google Drive hierarchy, data governance, and reference-image protocol |
| Appearance-Label Image Classifier | Image classification | Documentation and recovery path complete; execution blocked | Consented, documented image dataset with group-aware split metadata |
| Voice-Guided Grocery Cart Prototype | Speech and LLM application | Validation tests and syntax checks verified; live workflow not run | Microphone/audio runtime, user-provided `OPENAI_API_KEY`, authorised retailer accounts, and current site selectors |

## Standards applied during the revamp

For every project, the intended final standard is:

1. Preserve worthwhile original notebooks, scripts, images, and notes.
2. Add a clear final workflow, with EDA or inspection separated from model work when applicable.
3. Remove only true duplicate cells and unsafe evaluation patterns.
4. Document environment setup, data placement, execution commands, results, diagrams, limitations, and responsible use.
5. Verify all code locally when the required data and runtime dependencies are available.
6. Mark unverified projects honestly until their missing input is restored.

## Data placement convention

Where an original external dataset is needed, the revised project documentation
will use a local `data/` directory, ignored by Git when the data is too large or
restricted. It will never require a personal Google Drive path, a hard-coded Kaggle
runtime path, or an embedded secret.
