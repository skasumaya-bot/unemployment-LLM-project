\# LLM-Based Economic Policy Classification



\## A Proof-of-Concept for Unemployment Analysis



\### Overview



This project demonstrates how Large Language Model (LLM)-based Natural Language Processing (NLP) can be applied to economic policy analysis.



The project uses a zero-shot Transformer model to classify short economic policy statements into different policy categories and then maps those classifications to their potential direction of unemployment effects.



The project is designed as a proof-of-concept demonstrating the integration of:



\- Python

\- Natural Language Processing

\- Transformer-based language models

\- Economic policy classification

\- Quantitative model validation

\- Data visualization

\- Economic interpretation



\---



\## Research Motivation



Economic policy documents contain large amounts of textual information that can be difficult to classify manually.



Traditional empirical economics often relies on numerical indicators such as:



\- government expenditure

\- government revenue

\- interest rates

\- government debt

\- unemployment rates



However, policy decisions are also communicated through textual statements.



This project explores how LLM-based methods could help transform policy text into structured economic information that could potentially be used in quantitative economic research.



\---



\## Research Question



Can a zero-shot Transformer model classify economic policy statements into meaningful policy categories and identify their potential implications for unemployment?



\---



\## Policy Categories



The project uses six policy categories:



1\. Fiscal expansion

2\. Fiscal contraction

3\. Monetary easing

4\. Monetary tightening

5\. Labour support

6\. Neutral policy



\---



\## Methodology



The project follows a multi-stage workflow.



\### Step 1: Dataset Loading



A small demonstration dataset containing 10 policy statements is loaded using pandas.



Each observation contains:



\- ID

\- Date

\- Institution

\- Policy text

\- Human-labelled policy category

\- Human policy score



\### Step 2: Rule-Based Classification



A rule-based NLP classifier is created as a baseline.



The classifier identifies policy categories using predefined economic keywords and phrases.



\### Step 3: Policy Score



Policy categories are translated into a simple numerical policy score.



This provides a structured representation of the qualitative policy information.



\### Step 4: Zero-Shot Transformer Classification



The project uses the Hugging Face Transformers library and the BART-large-MNLI model.



The Transformer performs zero-shot classification without being specifically fine-tuned on the demonstration dataset.



\### Step 5: Error Analysis



Transformer predictions are compared with human-labelled categories.



Incorrect predictions are identified and analysed.



Model confidence scores are also examined.



\### Step 6: Unemployment Effect Index



Policy categories are mapped to a simplified expected unemployment effect:



| Policy Category | Expected Unemployment Effect | Score |

|---|---|---:|

| Fiscal expansion | Decrease | -1 |

| Fiscal contraction | Increase | +1 |

| Monetary easing | Decrease | -1 |

| Monetary tightening | Increase | +1 |

| Labour support | Decrease | -1 |

| Neutral | Neutral | 0 |



The resulting measure is called the:



\*\*Policy Unemployment Effect Index\*\*



This index is a conceptual demonstration rather than an estimated causal effect.



\### Step 7: Quantitative Validation



The Transformer is evaluated using:



\- Accuracy

\- Precision

\- Recall

\- Macro F1-score

\- Confusion matrix

\- Category-level performance



\### Step 8: Visualization



The project generates visual outputs showing:



\- Model accuracy comparison

\- Transformer confidence distribution

\- Confusion matrix

\- Unemployment-effect distribution

\- Confidence by policy category



\---



\## Demonstration Results



The demonstration dataset contains 10 observations.



\### Model Performance



| Measure | Rule-Based | Zero-Shot Transformer |

|---|---:|---:|

| Accuracy | 100.0% | 90.0% |

| Macro Precision | 1.000 | 0.944 |

| Macro Recall | 1.000 | 0.917 |

| Macro F1 | 1.000 | 0.911 |



The Transformer correctly classified 9 out of 10 observations.



The average Transformer confidence was approximately 0.857.



The unemployment-effect classification accuracy was 90%.



\---



\## Error Analysis



The Transformer incorrectly classified one observation involving government transfers.



The human-labelled category was:



`fiscal\_expansion`



while the Transformer predicted:



`fiscal\_contraction`



The prediction had a confidence score of approximately 0.589.



This illustrates why LLM-based classification should be validated against human-labelled data rather than being treated as automatically correct.



\---



\## Important Limitation



This project contains only 10 observations.



Therefore, the reported accuracy, precision, recall and F1-score should NOT be interpreted as evidence of general Transformer performance.



The project is intentionally designed as a small proof-of-concept demonstrating the research workflow.



The Policy Unemployment Effect Index is also a conceptual measure and should not be interpreted as a causal estimate of the effect of policy on unemployment.



\---



\## Potential Research Extension



A future research version could use real European policy documents from institutions such as governments, central banks and European institutions.



A possible extended research pipeline would be:



Policy documents

→ LLM classification

→ Policy stance

→ Policy score

→ Unemployment-effect score

→ Country-time panel

→ Econometric analysis



Such an extension could combine textual economic information with conventional macroeconomic variables and panel-data methods.



Potential applications include:



\- fiscal policy analysis

\- monetary policy analysis

\- labour-market policy analysis

\- economic shock analysis

\- unemployment forecasting

\- policy coordination measurement



\---



\## Technologies



\- Python

\- pandas

\- NumPy

\- scikit-learn

\- Matplotlib

\- Seaborn

\- Hugging Face Transformers

\- PyTorch

\- Excel



\---



\## Project Structure



```text

unemployment\_llm\_project/

│

├── unemployment\_llm\_project.py

├── unemployment\_policy\_text.csv.xlsx

│

├── unemployment\_policy\_llm\_final\_results.xlsx

├── unemployment\_llm\_validation\_results.xlsx

│

├── model\_accuracy\_comparison.png

├── transformer\_confidence\_distribution.png

├── transformer\_confusion\_matrix.png

├── policy\_unemployment\_effect\_distribution.png

├── transformer\_confidence\_by\_category.png

│

├── README.md

├── requirements.txt

└── .gitignore

