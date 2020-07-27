# Nub 1.0 Lesson Summarizer

## Model description
A [T5 Model](https://huggingface.co/soroush/t5-finetuned-lesson-summarizer) finetuned on lecture dataset + CNN/Daily Mail. 

>Students often struggle with processing large volumes of content throughout the course of their studies. This in turn will lead to a decrease in student productivity. Utilizing an effective summarization tool, students can better prepare for exams, improve their own summarization skills via learning by comparison, and avoid wasting time on low-quality content by skimming the summary. Nub 1.0 is a text summarizer that leverages deep learning to specialize in educational content. When evaluated on the CNN/Daily Mail dataset, It shows superior performance compared to similar tools. 


## Intended uses & limitations
### How to use Nub 1.0 
#### 1. Run the Web app locally (preferred method)
You'll need Python 3.7+, and pip. Simply clone the repo, and install the requirements:
```python
pip install -r  requirements.txt
```
Then start the web app from the command line:
```python
streamlit run app.py
```
This will print the local URL or open the web app in a broswer.

#### 2. Run the web app on Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-spIChqxJ4BAI2i70xAlMtEzxnJdcVPE?usp=sharing)

#### 3. Play around with the model under the hood
Install Transformers:
 ```python
pip install transformers
```
Load from huggingface [model hub](https://huggingface.co/soroush/t5-finetuned-lesson-summarizer):
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

To process the lectures from ML4T:
 ```
python process_srt_files.py \
    --output_dir './data/processed_lessons/' \
    --lessons_dir 'data/raw_lessons'
``` 
Each lesson is a collection of videos in a directory that lives in `raw_lessons`.
This script will take video transcripts in `.srt` format and putputs a single-line text document per lesson.

To run the Question Answering module to autogenerate unsupervised summaries for finetuning, run:
```
python run_qa.py \
    --output_dir './data/qa_generated_summaries/' 
    --lessons_dir './data/processed_lessons' 
    --questions_dir './data/lesson_questions'
```
- `lesson_questions`: directory containing question files (names match the lesson files)
- `processed_lessons`: the output of `process_srt_files.py`.  
- `qa_generated_summaries`: the directory containing the QA generated summaries
- `raw_lessons`: each directory inside here has one or more `.srt` files
 

## Training procedure
To fine-tune the model on a downstream task follow steps in `nub-training-evaluation/fine-tuning T-5 on CNN+daily mail + ML4T.ipynb`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sbassam/nub-summarizer/blob/master/fine_tuning_T_5_on_CNN%2Bdaily_mail_%2B_ML4T.ipynb)

## Eval results

Step by step process documented here: `nub-training-evaluation/Run Evaluations on Fine-tuned T-5 Summarizer.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sbassam/nub-summarizer/blob/master/Run_Evaluations_on_Fine_tuned_T_5_Summarizer.ipynb)

Comparative Analyses documented in `nub-training-evaluation/Results Analysis and Comparison.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sbassam/nub-summarizer/blob/master/Results_Analysis_and_Comparison.ipynb)
 
- Analysis result can be found in the corresponding csv files under `/nub-training-evaluation/result`
### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```

* readme guide from https://github.com/huggingface/model_card/edit/master/README.md

---
Tags
- [T5](https://github.com/huggingface/transformers)
- [LongFormer](https://github.com/huggingface/transformers)

License 
- [Apache License 2.0](https://github.com/sbassam/nub-summarizer/blob/master/LICENSE)

Datasets
- [CNN/Daily Mail](https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz) 
- [ML4T](https://www.udacity.com/course/machine-learning-for-trading--ud501)

Metrics
- [ROUGE-1, ROUGE-2, ROUGE-L](https://pypi.org/project/rouge-score/)
---

## 
