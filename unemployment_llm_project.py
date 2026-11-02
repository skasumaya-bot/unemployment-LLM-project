import pandas as pd
from pathlib import Path

from transformers import pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#       UNEMPLOYMENT POLICY LLM PROJECT
# ============================================================

print("\n" + "=" * 60)
print("       UNEMPLOYMENT POLICY LLM PROJECT")
print("=" * 60)


# ============================================================
# STEP 1: LOADING DATASET
# ============================================================

print("\nSTEP 1: LOADING DATASET")
print("-" * 60)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
file_name = PROJECT_DIR / "unemployment_policy_text.csv.xlsx"

df = pd.read_excel(file_name)
required_columns = {"id", "institution", "policy_text", "human_category"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
if df.empty:
    raise ValueError("The dataset must contain at least one observation.")
if df[list(required_columns)].isna().any().any():
    raise ValueError("Required columns must not contain missing values.")
if df["policy_text"].astype(str).str.strip().eq("").any():
    raise ValueError("Policy text must not be blank.")
POLICY_LABELS = [
    "fiscal_expansion", "fiscal_contraction", "monetary_easing",
    "monetary_tightening", "labour_support", "neutral"
]
invalid_labels = set(df["human_category"]) - set(POLICY_LABELS)
if invalid_labels:
    raise ValueError(f"Unsupported human labels: {sorted(invalid_labels)}")

print("Dataset loaded successfully.")
print(f"Number of observations: {len(df)}")

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# STEP 2: RULE-BASED CLASSIFICATION
# ============================================================

print("\nSTEP 2: RULE-BASED CLASSIFICATION")
print("-" * 60)


def classify_policy_rule_based(text):

    text = str(text).lower()

    if "interest rate" in text and "reduced" in text:
        return "monetary_easing"

    elif "interest rate" in text and "increased" in text:
        return "monetary_tightening"

    elif "wage subsidies" in text:
        return "labour_support"

    elif "employment subsidy" in text:
        return "labour_support"

    elif "public investment" in text:
        return "fiscal_expansion"

    elif "government transfers" in text:
        return "fiscal_expansion"

    elif "public expenditure" in text and "reduced" in text:
        return "fiscal_contraction"

    elif "spending" in text and "reduced" in text:
        return "fiscal_contraction"

    elif "no major changes" in text:
        return "neutral"

    elif "asset purchase" in text:
        return "monetary_easing"

    else:
        return "unknown"


df["rule_based_category"] = df["policy_text"].apply(
    classify_policy_rule_based
)


# Calculate rule-based accuracy

rule_accuracy = (
    df["rule_based_category"] == df["human_category"]
).mean()

print("\nRule-based classification completed.")
print(
    f"Rule-based accuracy: {rule_accuracy * 100:.1f}%"
)


# ============================================================
# STEP 3: POLICY SCORE
# ============================================================

print("\nSTEP 3: POLICY SCORE")
print("-" * 60)


def policy_effect_from_category(category):
    """Return a conceptual assumption, not an estimated unemployment effect."""

    if category in [
        "fiscal_expansion",
        "monetary_easing",
        "labour_support"
    ]:
        return "decrease"

    elif category in [
        "fiscal_contraction",
        "monetary_tightening"
    ]:
        return "increase"

    else:
        return "neutral"


def policy_score_from_category(category):

    if category in [
        "fiscal_expansion",
        "monetary_easing",
        "labour_support"
    ]:
        return -1

    elif category in [
        "fiscal_contraction",
        "monetary_tightening"
    ]:
        return 1

    else:
        return 0


df["human_assumed_direction"] = (
    df["human_category"].apply(
        policy_effect_from_category
    )
)

df["human_assumed_score"] = (
    df["human_category"].apply(
        policy_score_from_category
    )
)


print(
    "Directional scores are predefined assumptions, not independently "
    "validated unemployment predictions or estimated causal effects."
)


# ============================================================
# STEP 4: ZERO-SHOT TRANSFORMER CLASSIFICATION
# ============================================================

print("\nSTEP 4: ZERO-SHOT TRANSFORMER CLASSIFICATION")
print("-" * 60)

print("\nLoading zero-shot Transformer model...")
print("This may take some time on the first run.\n")


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


candidate_labels = [
    "fiscal expansion",
    "fiscal contraction",
    "monetary easing",
    "monetary tightening",
    "labour support",
    "neutral policy"
]


def classify_with_transformer(text):

    result = classifier(
        text,
        candidate_labels,
        multi_label=False
    )

    predicted_label = result["labels"][0]

    # Relative model score over the supplied labels; not calibrated correctness.
    confidence = result["scores"][0]

    label_mapping = {

        "fiscal expansion":
            "fiscal_expansion",

        "fiscal contraction":
            "fiscal_contraction",

        "monetary easing":
            "monetary_easing",

        "monetary tightening":
            "monetary_tightening",

        "labour support":
            "labour_support",

        "neutral policy":
            "neutral"
    }

    predicted_category = label_mapping[
        predicted_label
    ]

    return predicted_category, confidence


transformer_categories = []
transformer_confidences = []


print("Classifying policy texts with Transformer...\n")


for text in df["policy_text"]:

    category, confidence = (
        classify_with_transformer(text)
    )

    transformer_categories.append(category)

    transformer_confidences.append(confidence)


df["transformer_category"] = (
    transformer_categories
)

df["transformer_confidence"] = (
    transformer_confidences
)


# Transformer accuracy

transformer_accuracy = (
    df["transformer_category"]
    ==
    df["human_category"]
).mean()


average_confidence = (
    df["transformer_confidence"].mean()
)


print("\n")
print("=" * 60)
print("UNEMPLOYMENT POLICY CLASSIFICATION RESULTS")
print("=" * 60)


print(
    df[
        [
            "id",
            "institution",
            "human_category",
            "rule_based_category",
            "transformer_category",
            "transformer_confidence"
        ]
    ].to_string(index=False)
)


print("\n" + "-" * 60)
print("MODEL COMPARISON")
print("-" * 60)

print(
    f"Rule-based accuracy: "
    f"{rule_accuracy * 100:.1f}%"
)

print(
    f"Transformer accuracy: "
    f"{transformer_accuracy * 100:.1f}%"
)

print(
    f"Average Transformer confidence: "
    f"{average_confidence:.3f}"
)


# ============================================================
# STEP 5: TRANSFORMER ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: TRANSFORMER ERROR ANALYSIS")
print("=" * 60)


df["correct_prediction"] = (
    df["transformer_category"]
    ==
    df["human_category"]
)


# Confidence levels

def confidence_level(score):

    if score >= 0.80:
        return "High"

    elif score >= 0.60:
        return "Medium"

    else:
        return "Low"


df["confidence_level"] = (
    df["transformer_confidence"].apply(
        confidence_level
    )
)


incorrect_predictions = df[
    ~df["correct_prediction"]
]


print("\nIncorrect Transformer Predictions:")


if len(incorrect_predictions) == 0:

    print("No incorrect predictions.")

else:

    print(
        incorrect_predictions[
            [
                "id",
                "institution",
                "policy_text",
                "human_category",
                "transformer_category",
                "transformer_confidence"
            ]
        ].to_string(index=False)
    )


transformer_error_rate = (
    1 - transformer_accuracy
)


number_incorrect = (
    (~df["correct_prediction"]).sum()
)


number_correct = (
    df["correct_prediction"].sum()
)


print("\n" + "-" * 60)
print("ERROR ANALYSIS")
print("-" * 60)

print(
    f"Transformer error rate: "
    f"{transformer_error_rate * 100:.1f}%"
)

print(
    f"Number of incorrect predictions: "
    f"{number_incorrect}"
)

print(
    f"Number of correct predictions: "
    f"{number_correct}"
)


print("\n" + "-" * 60)
print("CONFIDENCE ANALYSIS")
print("-" * 60)

print(
    df["confidence_level"].value_counts()
)


# ============================================================
# STEP 6: ASSUMED DIRECTIONAL POLICY INDEX
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: ASSUMED DIRECTIONAL POLICY INDEX")
print("=" * 60)


df["transformer_assumed_direction"] = (
    df["transformer_category"].apply(
        policy_effect_from_category
    )
)


df["transformer_assumed_score"] = (
    df["transformer_category"].apply(
        policy_score_from_category
    )
)


df["effect_prediction_correct"] = (
    df["human_assumed_direction"]
    ==
    df["transformer_assumed_direction"]
)


effect_accuracy = (
    df["effect_prediction_correct"].mean()
)


# Assumed Directional Policy Index

assumed_directional_policy_index = (
    df["transformer_assumed_score"].mean()
)


print("\nPolicy Unemployment Effect Results:")

print(
    df[
        [
            "id",
            "human_category",
            "transformer_category",
            "human_assumed_direction",
            "transformer_assumed_direction",
            "human_assumed_score",
            "transformer_assumed_score",
            "effect_prediction_correct"
        ]
    ].to_string(index=False)
)


print("\n" + "-" * 60)
print("POLICY EFFECT PERFORMANCE")
print("-" * 60)

print(
    f"Mapped-direction agreement: "
    f"{effect_accuracy * 100:.1f}%"
)

print(
    f"Assumed Directional Policy Index: "
    f"{policy_unemployment_effect_index:.3f}"
)


print("\n" + "-" * 60)
print("ASSUMED DIRECTION DISTRIBUTION")
print("-" * 60)

print(
    df["transformer_assumed_direction"]
    .value_counts()
)


# ============================================================
# STEP 7: QUANTITATIVE MODEL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: QUANTITATIVE MODEL VALIDATION")
print("=" * 60)


y_true = df["human_category"]

y_pred = df["transformer_category"]


# Accuracy

validation_accuracy = accuracy_score(
    y_true,
    y_pred
)


# Macro metrics

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    labels=POLICY_LABELS,
    zero_division=0
)


macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    labels=POLICY_LABELS,
    zero_division=0
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    labels=POLICY_LABELS,
    zero_division=0
)


print("\nMODEL ACCURACY")
print("-" * 60)

print(
    f"Rule-based accuracy: "
    f"{rule_accuracy * 100:.1f}%"
)

print(
    f"Transformer accuracy: "
    f"{validation_accuracy * 100:.1f}%"
)


print("\nTRANSFORMER VALIDATION METRICS")
print("-" * 60)

print(
    f"Macro Precision: "
    f"{macro_precision:.3f}"
)

print(
    f"Macro Recall: "
    f"{macro_recall:.3f}"
)

print(
    f"Macro F1-score: "
    f"{macro_f1:.3f}"
)


# Classification report

report = classification_report(
    y_true,
    y_pred,
    labels=POLICY_LABELS,
    zero_division=0
)


print("\n" + "-" * 60)
print("DETAILED CLASSIFICATION REPORT")
print("-" * 60)

print(report)


# Confusion matrix

labels = POLICY_LABELS


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)


print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_" + label
        for label in labels
    ],
    columns=[
        "Predicted_" + label
        for label in labels
    ]
)


print(cm_df)


# Per-category performance

report_dict = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    labels=POLICY_LABELS,
    zero_division=0
)


category_results = []


for category in labels:

    category_results.append(
        {
            "category": category,
            "precision":
                report_dict[category]["precision"],
            "recall":
                report_dict[category]["recall"],
            "f1_score":
                report_dict[category]["f1-score"],
            "support":
                report_dict[category]["support"]
        }
    )


category_results_df = pd.DataFrame(
    category_results
)


print("\n" + "-" * 60)
print("PER-CATEGORY PERFORMANCE")
print("-" * 60)

print(
    category_results_df.to_string(
        index=False
    )
)


# Calculate baseline metrics from predictions. Include unknown predictions
# in the macro average so unsupported rule outputs are not silently dropped.
rule_metric_labels = POLICY_LABELS + (
    ["unknown"] if "unknown" in set(df["rule_based_category"]) else []
)
rule_precision = precision_score(
    y_true, df["rule_based_category"], labels=rule_metric_labels,
    average="macro", zero_division=0
)
rule_recall = recall_score(
    y_true, df["rule_based_category"], labels=rule_metric_labels,
    average="macro", zero_division=0
)
rule_f1 = f1_score(
    y_true, df["rule_based_category"], labels=rule_metric_labels,
    average="macro", zero_division=0
)

# Model performance comparison

comparison_df = pd.DataFrame(
    {
        "Model": [
            "Rule-based",
            "Zero-shot Transformer"
        ],

        "Accuracy": [
            rule_accuracy,
            validation_accuracy
        ],

        "Precision": [
            rule_precision,
            macro_precision
        ],

        "Recall": [
            rule_recall,
            macro_recall
        ],

        "F1_Score": [
            rule_f1,
            macro_f1
        ]
    }
)


print("\n" + "-" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("-" * 60)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 7 INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 INTERPRETATION")
print("=" * 60)


print(
    f"\nThe Transformer correctly classified "
    f"{number_correct} out of "
    f"{len(df)} policy observations."
)


print(
    f"The Transformer achieved a macro "
    f"F1-score of {macro_f1:.3f}."
)


print(
    "These metrics describe only the supplied demonstration dataset; "
    "they do not establish performance on unseen policy documents."
)


print(
    "\nIMPORTANT: This dataset contains only "
    f"{len(df)} observations."
)


print(
    "Therefore, these validation statistics "
    "should be interpreted as a demonstration "
    "rather than general evidence of model "
    "performance."
)


# ============================================================
# STEP 8: VISUALIZATION AND RESEARCH OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: VISUALIZATION AND RESEARCH OUTPUTS")
print("=" * 60)


# ------------------------------------------------------------
# 8.1 MODEL ACCURACY COMPARISON
# ------------------------------------------------------------

print("\nCreating model accuracy comparison...")


plt.figure(figsize=(8, 5))

models = [
    "Rule-based",
    "Zero-shot Transformer"
]

accuracies = [
    rule_accuracy * 100,
    validation_accuracy * 100
]


plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Model Accuracy Comparison"
)

plt.ylim(0, 110)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_accuracy_comparison.png",
    dpi=300
)

plt.close()

print(
    "Saved: model_accuracy_comparison.png"
)


# ------------------------------------------------------------
# 8.2 TRANSFORMER CONFIDENCE DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nCreating Transformer confidence distribution..."
)


plt.figure(figsize=(8, 5))


plt.hist(
    df["transformer_confidence"],
    bins=5
)


plt.xlabel(
    "Transformer Confidence"
)

plt.ylabel(
    "Number of Observations"
)

plt.title(
    "Transformer Confidence Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transformer_confidence_distribution.png",
    dpi=300
)

plt.close()

print(
    "Saved: transformer_confidence_distribution.png"
)


# ------------------------------------------------------------
# 8.3 CONFUSION MATRIX
# ------------------------------------------------------------

print(
    "\nCreating confusion matrix..."
)


plt.figure(figsize=(9, 7))


sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title(
    "Transformer Confusion Matrix"
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transformer_confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "Saved: transformer_confusion_matrix.png"
)


# ------------------------------------------------------------
# 8.4 UNEMPLOYMENT EFFECT DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nCreating unemployment effect distribution..."
)


effect_counts = (
    df["transformer_assumed_direction"]
    .value_counts()
)


plt.figure(figsize=(8, 5))


effect_counts.plot(
    kind="bar"
)


plt.xlabel(
    "Assumed Direction from Policy Category"
)

plt.ylabel(
    "Number of Policies"
)

plt.title(
    "Assumed Direction from Policy Category Distribution"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "policy_unemployment_effect_distribution.png",
    dpi=300
)

plt.close()

print(
    "Saved: policy_unemployment_effect_distribution.png"
)


# ------------------------------------------------------------
# 8.5 CONFIDENCE BY POLICY CATEGORY
# ------------------------------------------------------------

print(
    "\nCreating confidence by policy category..."
)


category_confidence = (
    df.groupby(
        "transformer_category"
    )["transformer_confidence"]
    .mean()
    .sort_values(
        ascending=False
    )
)


plt.figure(figsize=(9, 5))


category_confidence.plot(
    kind="bar"
)


plt.xlabel(
    "Policy Category"
)

plt.ylabel(
    "Average Transformer Confidence"
)

plt.title(
    "Transformer Confidence by Policy Category"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transformer_confidence_by_category.png",
    dpi=300
)

plt.close()

print(
    "Saved: transformer_confidence_by_category.png"
)


# ============================================================
# STEP 8 RESEARCH SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 RESEARCH SUMMARY")
print("=" * 60)


print(
    f"\nRule-based classification accuracy: "
    f"{rule_accuracy * 100:.1f}%"
)


print(
    f"Zero-shot Transformer accuracy: "
    f"{validation_accuracy * 100:.1f}%"
)


print(
    f"Average Transformer confidence: "
    f"{average_confidence:.3f}"
)


print(
    f"Macro Precision: "
    f"{macro_precision:.3f}"
)


print(
    f"Macro Recall: "
    f"{macro_recall:.3f}"
)


print(
    f"Macro F1-score: "
    f"{macro_f1:.3f}"
)


print(
    f"Mapped-direction agreement: "
    f"{effect_accuracy * 100:.1f}%"
)


print(
    f"Assumed Directional Policy Index: "
    f"{policy_unemployment_effect_index:.3f}"
)


# ============================================================
# SAVE FINAL EXCEL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVING PROJECT RESULTS")
print("=" * 60)


# Save complete observation-level results

final_output_file = (
    "unemployment_policy_llm_final_results.xlsx"
)


df.to_excel(
    OUTPUT_DIR / final_output_file,
    index=False
)


# Save validation results

validation_output_file = (
    "unemployment_llm_validation_results.xlsx"
)


with pd.ExcelWriter(
    OUTPUT_DIR / validation_output_file,
    engine="openpyxl"
) as writer:

    comparison_df.to_excel(
        writer,
        sheet_name="Model_Comparison",
        index=False
    )

    category_results_df.to_excel(
        writer,
        sheet_name="Category_Performance",
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion_Matrix"
    )

    incorrect_predictions.to_excel(
        writer,
        sheet_name="Errors",
        index=False
    )


print("\n--------------------------------------------")
print("PROJECT COMPLETED SUCCESSFULLY.")
print("--------------------------------------------")


print(
    "\nFinal results saved to:"
)

print(
    final_output_file
)


print(
    "\nValidation results saved to:"
)

print(
    validation_output_file
)


print(
    "\nVisualization files created:"
)

print(
    "1. model_accuracy_comparison.png"
)

print(
    "2. transformer_confidence_distribution.png"
)

print(
    "3. transformer_confusion_matrix.png"
)

print(
    "4. policy_unemployment_effect_distribution.png"
)

print(
    "5. transformer_confidence_by_category.png"
)


print("\n" + "=" * 60)
print("END OF UNEMPLOYMENT POLICY LLM PROJECT")
print("=" * 60)