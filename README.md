# LLM-Based Economic Policy Classification
## A proof of concept for unemployment research

### Overview
This Python project demonstrates economic-policy text classification using a rule-based baseline and a zero-shot Transformer model (`facebook/bart-large-mnli`). Ten demonstration policy statements are classified into six categories, and predictions are compared against human-provided labels. An illustrative policy-unemployment-effect score is also constructed.

**Scope:** This is a small methodological proof of concept, not a validated forecasting model or an estimate of policy effects on unemployment.

### Research question
Can a zero-shot Transformer classify short economic-policy statements into policy categories, and how do its classifications compare with a rule-based baseline on a small labelled example?

### Policy categories
1. Fiscal expansion
2. Fiscal contraction
3. Monetary easing
4. Monetary tightening
5. Labour support
6. Neutral policy

### Methods
1. Read the demonstration spreadsheet with policy text and manually assigned reference categories.
2. Classify texts with a keyword-based rule system.
3. Apply the BART-large-MNLI zero-shot classifier to the same statements.
4. Compare predictions with reference labels using accuracy, precision, recall, macro F1-score and a confusion matrix.
5. Review misclassified statements and model confidence scores.
6. Map predicted categories to an illustrative, directional unemployment-effect score.
7. Export tables and generate summary figures.

### Demonstration results
| Measure | Rule-based | Zero-shot BART |
|---|---:|---:|
| Accuracy | 100.0% | 90.0% |
| Macro precision | 1.000 | 0.944 |
| Macro recall | 1.000 | 0.917 |
| Macro F1-score | 1.000 | 0.911 |

The documented example contains **10 statements**. The zero-shot model correctly classified 9 of 10; its average confidence was approximately 0.857. One statement concerning government transfers was classified as fiscal contraction rather than the reference category of fiscal expansion. These values are descriptive of the small demonstration only.

The expected-direction score assigns -1 to fiscal expansion, monetary easing and labour support; +1 to fiscal contraction and monetary tightening; and 0 to neutral policy. **It is an illustrative coding rule, not an econometric estimate or a causal effect.**

### Main repository files
- `unemployment_llm_project.py` — classification, validation, scoring, tables and figures
- `unemployment_policy_text.csv.xlsx` — demonstration input spreadsheet
- `unemployment_policy_llm_final_results.xlsx` — saved observation-level results
- `unemployment_llm_validation_results.xlsx` — saved comparison and validation tables
- `model_accuracy_comparison.png` — comparison chart
- `transformer_confusion_matrix.png` — prediction-error visualization
- Additional spreadsheets and figures document intermediate results and model confidence

### Reproduction
Install Python and the required libraries: `pandas`, `numpy`, `transformers`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`.

From the repository directory, run:

```bash
python unemployment_llm_project.py
```

The first execution may require downloading the Hugging Face `facebook/bart-large-mnli` model. Ensure that the input spreadsheet is present in the working directory. The script writes Excel output files and figures to the current working directory.

### Limitations
- The sample has only ten statements; metrics should not be generalized to unseen policy documents.
- The human-labelled categories are illustrative rather than a large adjudicated annotation dataset.
- Scores reflect assumed directional relationships, not observed labour-market outcomes.
- Model confidence does not imply calibrated probability or causal certainty.
- Future work could use a larger real policy corpus with careful validation, provenance and sensitivity analysis.

### Tools
Python, pandas, NumPy, Hugging Face Transformers, PyTorch, scikit-learn, Matplotlib, seaborn and Excel.
