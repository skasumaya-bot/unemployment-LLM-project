# Economic Policy Text Classification

## A Python proof of concept for unemployment-related policy analysis

This project compares a keyword baseline with the zero-shot Transformer
`facebook/bart-large-mnli`. It turns short policy statements into six categories:
fiscal expansion, fiscal contraction, monetary easing, monetary tightening,
labour support, and neutral policy.

It demonstrates Python data processing, NLP, classification evaluation,
error analysis, Excel export, and visualization. The model is used for
zero-shot classification, without fine-tuning on this dataset.

## Run the project

Use Python 3.11 or 3.12 and an isolated environment. From the repository folder:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies and run:

```bash
python -m pip install -r requirements.txt
python unemployment_llm_project.py
```

The first run requires internet access to download the pretrained model.
CPU execution can be slow. Dependencies are constrained by major versions,
but this is not an exact environment lockfile.

The script finds the input relative to its own location. New Excel workbooks
and five PNG charts are written to `outputs/`. Existing files in that folder
are overwritten on subsequent runs.

## Input

`unemployment_policy_text.csv.xlsx` is an Excel workbook despite its filename.
Required columns are `id`, `institution`, `policy_text`, and `human_category`.
The script rejects missing required values, empty data, blank texts, and
unsupported human categories.

The supplied dataset has 10 demonstration observations. Institutional names
alone do not verify provenance: real-document extensions should include
source URLs, dates, exact quotations, and an annotation guide.

## Workflow and validation

1. Load and check the demonstration dataset.
2. Classify text with explicit keyword rules.
3. Classify the same text with BART-large-MNLI and six candidate labels.
4. Compare both sets of predictions with human category labels.
5. Calculate accuracy and macro precision, recall, and F1 from predictions.
6. Examine errors, the confusion matrix, and relative model scores.
7. Map categories to an explicitly assumed directional score.
8. Export results and charts.

Transformer macro metrics use all six policy categories. Baseline macro
metrics additionally include `unknown` if the rules return it, penalizing
unsupported predictions. Macro averages give each included class equal weight;
undefined precision or recall is set to zero.

Model confidence is a relative score over the candidate labels. It is not a
calibrated probability that a prediction is correct. High/medium/low thresholds
in the script are descriptive choices.

## Assumed directional score

| Classified policy | Assumed unemployment direction | Score |
|---|---|---:|
| Fiscal expansion | Decrease | -1 |
| Fiscal contraction | Increase | +1 |
| Monetary easing | Decrease | -1 |
| Monetary tightening | Increase | +1 |
| Labour support | Decrease | -1 |
| Neutral | Neutral | 0 |

The **Assumed Directional Policy Index** is the mean of these assigned scores.
It summarizes category composition under these assumptions. It does not
estimate unemployment changes, forecast unemployment, or identify causal effects.
Actual effects may depend on timing, shock conditions, institutions, and policy
responses to unemployment.

**Mapped-direction agreement** compares the mapping of predicted categories
with the mapping of human categories. It is a coarser classification agreement
measure, not independent validation against observed unemployment outcomes.
The former self-comparison called “policy score accuracy” has been removed.

## Historical demonstration results

The original README reported these results on the 10 observations:

| Measure | Rule-based | Transformer |
|---|---:|---:|
| Accuracy | 100% | 90% |
| Macro precision | Not independently verified | 0.944 |
| Macro recall | Not independently verified | 0.917 |
| Macro F1 | Not independently verified | 0.911 |

The original baseline precision, recall, and F1 were hard-coded as 1.0.
The corrected script now computes them. The reported Transformer error involved
government transfers classified as fiscal contraction instead of fiscal expansion.

These figures are historical reports, not a fresh execution of the corrected
script. Root-level Excel results and PNGs are legacy artifacts; regenerated
results belong in `outputs/` and use the corrected terminology. Output filenames
retain their original names for continuity, while direction-related columns
now use `human_assumed_*` and `transformer_assumed_*`.

## Limitations and next research steps

Ten short statements cannot establish general model performance. The baseline
rules may fit the wording of the demonstration examples. Human labels require
a documented annotation process; this repository does not establish independent
annotation or inter-annotator agreement.

A research extension should collect a larger sourced corpus, define annotation
rules, obtain independent human labels, and reserve an untouched evaluation set.
Include paraphrases, negation, mixed policies, and ambiguous statements. Compare
models on the same held-out observations and report class counts and uncertainty.

Only after that validation should text-derived policy indicators be linked to
country-time data for econometric analysis. Such analysis needs a separate
identification strategy for claims about unemployment effects.

## Repository structure

- `unemployment_llm_project.py`: classification, evaluation, and export.
- `unemployment_policy_text.csv.xlsx`: supplied demonstration input.
- `requirements.txt`: Python dependencies.
- `.gitignore`: excludes local environments, generated outputs, and Excel lock files.
- `outputs/`: created when the script runs.
- Root-level PNG and result workbooks: historical demonstration artifacts.
