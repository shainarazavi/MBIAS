"""
Script to calculate Deepeval metrics for biased text within test data.
Taken from https://github.com/VectorInstitute/NewsMediaBias/blob/main/UnBIAS-Upgrade/Evaluation/deep_eval_original_text.py
"""
import pandas as pd
from deepeval.metrics import (
	BiasMetric,
	ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

USER = "username"

INPUT_FILE_PATH = (
    f'/h/{USER}/NewsMediaBias/UnBIAS-Library/datasets/train_500.csv'
)

OUTPUT_PATH = f"/h/{USER}/NewsMediaBias/mbias/eval/train-500-biased-evaluation.csv"

MODEL = "gpt-3.5-turbo"

#Initialize metrics to be calculated
bias_metric = BiasMetric(threshold=0.0, model=MODEL)
toxicity_metric = ToxicityMetric(threshold=0.0, model=MODEL)

#Read Orginal test file for evaluation
df = pd.read_csv(INPUT_FILE_PATH)
METRIC_COLS = ["Original_Bias_Score", "Original_Toxicity_Score"]
for col in METRIC_COLS:
	df[col] = -1.0

#Get DeepEval scores for every row in test data 
for index in range(len(df)):
	print(f"Calculating scores for test case {index}.")
	input_text = df.loc[index, "biased_text"]

	test_case = LLMTestCase(input=input_text, actual_output=input_text)

	try:
		bias_metric.measure(test_case)
		df.loc[index, "Original_Bias_Score"] = bias_metric.score
	except ValueError:
		print(f"ValueError when calculating Bias metric for the test case {index}.")

	try:
		toxicity_metric.measure(test_case)
		df.loc[index, "Original_Toxicity_Score"] = toxicity_metric.score
	except ValueError:
		print(f"ValueError when calculating Toxicity metric for the test case {index}.")

#Get average of valid test cases.
for col in METRIC_COLS:
	average = df.loc[df[col] != -1, col].mean()
	count_negative_ones = (df[col] == -1).sum()
	print(f"Average of {col}: {average}. ({count_negative_ones}/{len(df)}) invalid.")

#Store evaluation results in a file
df.to_csv(OUTPUT_PATH)
