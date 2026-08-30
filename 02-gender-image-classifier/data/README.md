# Data recovery contract

The original image dataset is not redistributed with this project. The tracked
training_log_2022.csv file is a transcription of the original notebook’s saved
training output, not image data.

Do not place arbitrary images here merely to make the notebook run. A legitimate
research rerun needs written permission to use every image and a dataset card
that defines what the two folder labels mean.

## Required local structure

~~~text
data/
├── class_a/                  # source-defined label 0
├── class_b/                  # source-defined label 1
├── split_manifest.csv        # image, label, person_id, source_id, session_id, split
├── dataset_card.md           # provenance, licence, consent, scope, known gaps
└── training_log_2022.csv     # tracked historical evidence; not a dataset
~~~

The historical notebook expects a single directory with two immediate class
subfolders. A future implementation should create training, validation, and test
directories from split_manifest.csv after assigning each person and related
images to exactly one partition.

## Minimum data-card fields

| Field | Why it is necessary |
|---|---|
| Source and licence | Establishes the right to train, evaluate, and publish. |
| Consent and allowed purpose | Prevents reuse beyond the contributor’s agreement. |
| Label definition and mapping | Avoids treating an ambiguous folder name as identity. |
| Person/source/session ID | Prevents identity and near-duplicate leakage. |
| Collection conditions | Reveals camera, lighting, background, and source shortcuts. |
| Exclusions and coverage limits | Makes missing populations and failure risks explicit. |
| Retention/removal process | Supports privacy and governance obligations. |

Do not commit raw images, personally identifying metadata, trained weights, or
model outputs without a specific rights and privacy review.
