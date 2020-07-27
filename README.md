---
language: `multilingual`
thumbnail: https://huggingface.co/soroush/t5-finetuned-lesson-summarizer
tags:
- T5
- LongFormer
- tags
license: "any valid license identifier"
datasets:
- [CNN/Daily Mail] [ML4T]
metrics:
- [ROUGE-1, ROUGE-2, ROUGE-L]
---

# Nub 1.0 Lesson Summarizer

## Model description
T5 finetuned on lecture dataset + CNN/Daily Mail.
https://huggingface.co/soroush/t5-finetuned-lesson-summarizer

## Intended uses & limitations
The main goal for this summarizer is to be used for learning content summarization
#### How to use Nub 1.0 Web app
You'll need Python 3.7+, and pip. Simply install the requirements:
```python
pip install requirements.txt
```
then from command line:
```python
streamlit run app.py
```
This will open the web app in the broswer.

#### How to use the underlying finetuned T5 model
 - load from `/nub-training-evaluation/model`
 - load from huggingface model hub:
 ```python
pip install transformers
```
 ```.python
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained('soroush/t5-finetuned-lesson-summarizer')
tokenizer = T5Tokenizer.from_pretrained('soroush/t5-finetuned-lesson-summarizer')
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
```
Now enter your text and pass it to the pipeline:
```python
input_text = '''Up until this point, we've focused on learners that provide forecast price changes. We then buy or sell the stocks with the most significant predicted price change. This approach ignores some important issues, such as the certainty of the price change. It also doesn't help us know when to exit the position either. In this lesson, we'll look at reinforcement learning. Reinforcement learners create policies that provide specific direction on which action to take. It's important to point out that when we say reinforcement learning, we're really describing a problem, not a solution. In the same way that linear regression is one solution to the supervised regression problem, there are many algorithms that solve the RL problem. Because I started out as a roboticist, I'm going to first explain this in terms of a problem for a robot. '''
output = summarizer(input_text)
output_text = output[0]['summary_text']
print(output_text)
```
#### Limitations and bias

The summarizer might take a long time with longer documents. Dividing the document into a few chunks and running them separately helps with the running time.
When `run_qa.py` you may need to decrease the `max_length=4096` to match the hardware capacity.

If installing the pytorch version is too heavy for the hardware in use, try downgrading to 
a cpu-only version `pip install requirements_cpu.txt`
## Training data

CNN/Daily Mail news dataset
To process the lectures from ML4T:
 ```
python process_srt_files.py \
    --output_dir './data/processed_lessons/' \
    --lessons_dir 'data/raw_lessons'
``` 
To run the Question Answering module to autogenerate unsupervised summaries.
```
python run_qa.py \
    --output_dir './data/qa_generated_summaries/' 
    --lessons_dir './data/processed_lessons' 
    --questions_dir './data/lesson_questions'
```
Use `/nub-training-evaluation/fine-tuning T-5 on CNN+daily mail + ML4T.ipynb` instead of `finetune_t5.py` to speed up he process.
## Training procedure
Training is under `/nub-training-evaluation`. Everything is documented in the corresponding Colab notebook.
Step by step process documented here `fine-tuning T-5 on CNN+daily mail + ML4T.ipynb`

## Eval results
raining is under `/nub-training-evaluation`.
Step by step process documented here `Run Evaluations on Fine-tuned T-5 Summarizer.ipynb`
Comparative Analyses documented in `Results Analysis and Comparison.ipynb` 
Analysis result can be found in the corresponding csv files under `/nub-training-evaluation/result`
### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```

* readme guide from https://github.com/huggingface/model_card/edit/master/README.md

## 
