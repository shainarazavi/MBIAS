
"""
Script to calculate Deepeval metrics for biased text within test data.
Taken from https://github.com/VectorInstitute/NewsMediaBias/blob/main/UnBIAS-Upgrade/Evaluation/deep_eval_unbiased_text.py
"""
import pandas as pd
import openai
from deepeval.metrics import (
	BiasMetric,
	KnowledgeRetentionMetric,
	ToxicityMetric,
	FaithfulnessMetric,
	AnswerRelevancyMetric
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase

USER = "username"
EVAL_FILE_PATH = (
	f"/h/{USER}/NewsMediaBias/mbias/results/mbias-inference-answers.csv"
)
OUTPUT_PATH = (
    f"/h/{USER}/NewsMediaBias/mbias/eval/mbias-evaluation.csv"
)

MODEL = "gpt-3.5-turbo"
DEBUG = False 

#Initialize metrics to be calculated
bias_metric = BiasMetric(threshold=0.0, model=MODEL, include_reason=True)
toxicity_metric = ToxicityMetric(threshold=0.0, model=MODEL)
knowledge_retention_metric = KnowledgeRetentionMetric(threshold=1, model=MODEL)
faithfulness_metric = FaithfulnessMetric(threshold=1, model=MODEL, include_reason=True)
answerrelevancy_metric = AnswerRelevancyMetric(threshold=1, model=MODEL, include_reason=True)

#Read Inference file for evaluation
df = pd.read_csv(EVAL_FILE_PATH)
METRIC_COLS = ["Bias_Score", "Toxicity_Score", "Knowledge_Retention_Score", "Faithfulness_Score",  "AnswerRelevancy_Score"] 
REASON_COLS = ["Bias_Reasons", "Knowledge_Retention_Reasons", "Faithfulness_Reasons", "AnswerRelevancy_Reasons"]

#Initialize all score columns to -1.0
for col in METRIC_COLS:
	df[col] = -1.0
for col in REASON_COLS:
	df[col] = ""

#Get DeepEval scores for every row in inference file 
for index in range(len(df)):

	print(f"Calculating scores for test case {index}.")

	input_text = df.loc[index, "biased_text"]
	model_output_text = df.loc[index, "debiased_text"]
	retrieval_context = [df.loc[index, "biased_text"]]

	test_case_1 = LLMTestCase(input=input_text, actual_output=model_output_text, retrieval_context=retrieval_context)
	messages = [
		LLMTestCase(
			input="",
			actual_output="Hello! I'm here to assist you debias your input text. Can you give me your text please",
		),
		LLMTestCase(input=input_text, actual_output=model_output_text),
	]
	test_case2 = ConversationalTestCase(messages=messages)

	try:
		bias_metric.measure(test_case_1)
		df.loc[index, "Bias_Score"] = bias_metric.score
		df.loc[index, "Bias_Reasons"] = bias_metric.reason
	except ValueError:
		print(f"ValueError when calculating Bias metric for the test case {index}.")
	except openai.BadRequestError:
		print(f"openai.BadRequestError when calculating Bias metric for the test case {index}.")
	except Exception as e:
		print("Exception while calculating Bias metric {}".format(str(e)))
		
	try:
		toxicity_metric.measure(test_case_1)
		df.loc[index, "Toxicity_Score"] = toxicity_metric.score
	except ValueError:
		print(f"ValueError when calculating Toxicity metric for the test case {index}.")
	except openai.BadRequestError:
		print(f"openai.BadRequestError when calculating Toxicity metric for test case {index}.")
	except KeyError:
		print(f"KeyError when calculating Toxicity metric for test case {index}.")

	try:
		knowledge_retention_metric.measure(test_case2)
		df.loc[index, "Knowledge_Retention_Score"] = knowledge_retention_metric.score
		df.loc[index, "Knowledge_Retention_Reasons"] = knowledge_retention_metric.reason
	except ValueError:
		print(
			f"ValueError when calculating Knowledge Retention metric for the test case {index}."
		)
	except openai.BadRequestError:
		print(f"openai.BadRequestError when calculating Knowledge Retention metric for test case {index}.")
	except Exception as e:
		print("Exception while calculating Knowledge Retention Metric: {}".format(str(e)))

	try:
		faithfulness_metric.measure(test_case_1)
		df.loc[index, "Faithfulness_Score"] = faithfulness_metric.score
		df.loc[index, "Faithfulness_Reasons"] = faithfulness_metric.reason
	except ValueError:
		print(
			f"ValueError when calculating Faithfulness metric for the test case {index}."
		)
	except openai.BadRequestError:
		print(f"openai.BadRequestError when calculating Faithfulness metric for test case {index}.")

	except Exception as e:
		print("Exception while calculating Faithfulness metric {}".format(str(e)))
	try:
		answerrelevancy_metric.measure(test_case_1)
		df.loc[index, "AnswerRelevancy_Score"] = answerrelevancy_metric.score
		df.loc[index, "AnswerRelevancy_Reasons"] = answerrelevancy_metric.reason
	except ValueError:
		print(
			f"ValueError when calculating AnswerRelevancy metric for the test case {index}."
		)
	except openai.BadRequestError:
		print(f"openai.BadRequestError when calculating AnswerRelevancy metric for test case {index}.")
	except Exception as e:
		print("Exception as {}".format(str(e)))

	if DEBUG:
		print("Bias")
		print(bias_metric.score)
		print(bias_metric.reason)

		print("Toxicity")
		print(toxicity_metric.score)

		print("Knowledge Retention")
		print(knowledge_retention_metric.score)
		print(knowledge_retention_metric.reason)

		print("Faithfulness Metric")
		print(faithfulness_metric.score)
		print(faithfulness_metric.reason)

		print("Answer Relevancy")
		print(answerrelevancy_metric.score)
		print(answerrelevancy_metric.reason)

#Get average of valid test cases.
for col in METRIC_COLS:
	average = df.loc[df[col] != -1, col].mean()
	count_negative_ones = (df[col] == -1).sum()
	print(f"Average of {col}: {average}. ({count_negative_ones}/{len(df)}) invalid.")


df.to_csv(OUTPUT_PATH)
