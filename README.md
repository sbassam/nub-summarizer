---
language: `multilingual`
thumbnail: "url to a thumbnail used in social sharing"
tags:
- array
- of
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
#### How to use
You'll need Python 3.7+, and pip. Simply install the requirements:
```python
pip install requirements.txt
```
then from command line:
```python
streamlit run app.py
```
This will open the web app in the broswer.


#### Limitations and bias

The summarizer might take a long time with longer documents. Dividing the document into a few chunks and running them separately helps with the running time.
When `run_qa.py` you may need to decrease the `max_length=4096` to match the hardware capacity.

## Training data

CNN/Daily Mail news dataset
To process the lectures from ML4T:
 ```
python process_srt_files.py \
    --output_dir './data/processed_lessons/' \
    --lessons_dir 'data/raw_lessons'
``` 
To run the Question Answering module to autogenerate unsupervised summaries
```
python run_qa.py \
    --output_dir './data/qa_generated_summaries/' 
    --lessons_dir './data/processed_lessons' 
    --questions_dir './data/lesson_questions'
```

## Training procedure

Step by step process documented here `fine-tuning T-5 on CNN+daily mail + ML4T.ipynb`

## Eval results
Step by step process documented here `Run Evaluations on Fine-tuned T-5 Summarizer.ipynb`
### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```

* readme guide from https://github.com/huggingface/model_card/edit/master/README.md

## 
