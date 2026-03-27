---
license: apache-2.0
task_categories:
- question-answering
language:
- en
tags:
- math
- synthetic
size_categories:
- 100K<n<1M
---

This dataset is provided to facilitate access to **GSM8k-Aug**, originally from https://github.com/da03/Internalize_CoT_Step_by_Step and https://arxiv.org/pdf/2405.14838.



**This dataset is used to train CODI (https://arxiv.org/abs/2502.21074)**


**Description**: 
*We utilize two datasets to train our models--GSM8k-Aug and GSM8k-Aug-NL. (1) We use the GSM8k-Aug dataset, which has proven effective for training implicit CoT methods. This dataset extends the original GSM8k training set to 385k samples by prompting GPT-4. To facilitate implicit CoT training, all natural language interleaving within the CoT is removed, leaving only structured mathematical expressions such as “<<10/5=2>> <<2*2=4>> <<6*4=24>>”. (2) We also use GSM8k-Aug-NL, a version that preserves natural language explanations, to assess both the generalizability and effectiveness of our approach to compress more verbose CoTs.*


**Data Format**:
\<question\>||\<CoT\>####\<Answer\>