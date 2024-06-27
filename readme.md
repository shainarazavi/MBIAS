# MBIAS

## License
This project is licensed under the MIT License.

## Datasets
This model utilizes the following dataset:
https://huggingface.co/datasets/newsmediabias/Bias-DeBiased

The model weights are here:
https://huggingface.co/newsmediabias/MBIAS

## Metrics
The primary performance metric for this model is accuracy.

## Pipeline Tag
`text-generation`

## Model Details
**Model Name:** MBIAS  
**Model Type:** Large Language Model (LLM)  
**Version:** 1.0  

## Model Description
MBIAS is a fine-tuned Large Language Model specifically designed to enhance safety while retaining contextual accuracy in model outputs. Traditional safety interventions often compromise contextual meaning when mitigating bias and toxicity. MBIAS addresses this by maintaining high contextual relevance and drastically reducing bias and toxicity in text generation.

## Intended Use
The model is intended for research and development purposes, particularly in applications where reducing bias and toxicity in language generation is crucial without sacrificing the retention of key information.

## Training Data
The model was fine-tuned on a custom dataset curated for comprehensive safety interventions. This dataset includes diverse text samples aiming to cover a wide range of demographics to effectively test and reduce bias and toxicity.

## Evaluation
MBIAS has demonstrated a significant reduction in bias and toxicity, with over 30% reduction overall and exceeding 90% in specific demographic analysis on an out-of-distribution test set. Performance metrics include bias reduction, toxicity reduction, and retention of key information (KR).

## How to Use
The model can be accessed and used for text generation through the HuggingFace platform. For detailed usage, please refer to the provided link in the model repository.

## Hyperparameters
- **Batch Size per GPU:** Training: 8, Evaluation: 4
- **Steps to Accumulate Gradients:** 1
- **Maximum Gradient Norm:** 0.3
- **Initial Learning Rate:** 2e-05
- **Weight Decay:** 0.001
- **Optimizer:** paged_adamw 8bit
- **Learning Rate Scheduler:** Constant
- **Warmup Steps Ratio:** 0.05
- **Maximum Sequence Length:** 2048
- **Training Epochs:** 2
- **LoRA Attention Dimension:** 64
- **LoRA Scaling/Dropout Probability:** 16/0.2

## Contact
For more information or questions, please contact Shaina Raza at [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai).
