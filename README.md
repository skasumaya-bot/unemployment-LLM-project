# Economic Policy Classification with Python and AI

This project explores how Python and an AI language model can classify short economic policy statements. It connects economic policy analysis with practical text analysis.

## What does the project do?

The project reads 10 example policy statements from an Excel file and classifies each statement in two ways:

- **Keyword rules:** look for phrases such as “interest rate” or “public investment.”
- **AI model:** use the pretrained BART-large-MNLI model to choose a policy category.

It compares these classifications with human labels, checks mistakes, and saves the results in Excel and charts. The AI model is used as provided, without additional training on these examples.

## Policy categories

| Category | Meaning |
|---|---|
| Fiscal expansion | Government increases spending or transfers |
| Fiscal contraction | Government reduces spending |
| Monetary easing | Central bank lowers interest rates or purchases assets |
| Monetary tightening | Central bank raises interest rates |
| Labour support | Measures such as wage or employment subsidies |
| Neutral policy | No major policy change |

## Demonstration results

The original demonstration reported:

| Method | Correct classifications | Accuracy |
|---|---:|---:|
| Keyword rules | 10 out of 10 | 100% |
| AI model | 9 out of 10 | 90% |

The AI model classified one statement about government transfers as fiscal contraction instead of fiscal expansion.

These results come from the earlier demonstration. The corrected code has not yet been run again to update the results and charts.

## What does the unemployment score mean?

The project assigns a simple score to each policy category:

| Policy categories | Assumed direction of unemployment | Score |
|---|---|---:|
| Fiscal expansion, monetary easing, labour support | Decrease | -1 |
| Fiscal contraction, monetary tightening | Increase | +1 |
| Neutral policy | No change | 0 |

These directions are assumptions used for this exercise. They are not measured effects on unemployment. The average score summarizes the types of policies in the dataset.

## How to run the project

1. Download this repository: select **Code → Download ZIP** on GitHub, then extract the folder.
2. Use Python 3.11 or 3.12 for this project.
3. Open a terminal in the extracted project folder.
4. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the script:

   ```bash
   python unemployment_llm_project.py
   ```

Keep the supplied Excel dataset in the same folder as the Python script. The first run needs internet access to download the AI model and may take some time.

## Main files

| File or folder | Purpose |
|---|---|
| `unemployment_llm_project.py` | Runs the analysis |
| `unemployment_policy_text.csv.xlsx` | Contains the 10 example statements and human labels |
| `requirements.txt` | Lists the Python packages needed |
| `outputs/` | Stores new Excel results and five charts when the script runs |

The input is an Excel file despite having “csv” in its name. Charts and result workbooks already in the main project folder are from the earlier demonstration. Running the script again replaces files in `outputs/`.

## Limitations and next step

This is a small learning and portfolio project. Ten examples are not enough to show how well either method works on other policy documents. The AI model's confidence score also does not guarantee that its classification is correct.

The next step is to add more policy statements with clear sources and independently checked human labels, then evaluate the methods on examples that were not used to develop the rules.
