import streamlit as st
import time
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_auto import AutoModelWithLMHead
# use this for heroku to avoid slug size issue
## torch==1.5.1
# torchvision==0.6.1

def main():
    st.title('Nub 1.0 Summarizer')
    input_text = st.text_area(label="Enter text", height=300, value="Up until this point, we've focused on learners that provide forecast price changes. We then buy or sell the stocks with the most significant predicted price change. This approach ignores some important issues, such as the certainty of the price change. It also doesn't help us know when to exit the position either. In this lesson, we'll look at reinforcement learning. Reinforcement learners create policies that provide specific direction on which action to take. It's important to point out that when we say reinforcement learning, we're really describing a problem, not a solution. In the same way that linear regression is one solution to the supervised regression problem, there are many algorithms that solve the RL problem. Because I started out as a roboticist, I'm going to first explain this in terms of a problem for a robot. ")
    'or...'
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    status = st.empty()

    with st.spinner('Downloading the model...'):
        tokenizer, model = load_model()
    status.success('Done!')
    status.empty()

    if st.button('Summarize'):
        if uploaded_file and not input_text:
            text_from_file = process_file(uploaded_file)
            run_summarizer(text_from_file, tokenizer, model)
        else:
            if input_text:
                run_summarizer(input_text, tokenizer, model)
            else:
                'gotta give *some* input'
    #todo: add radio button




def process_file(uploaded_file):
    input_text = uploaded_file.getvalue()

    # uploaded_file = io.StringIO(uploaded_file)
    # input_text = open(uploaded_file, "r")

    # Retrieve file contents -- this will be
    # 'First line.\nSecond line.\n'
    # input_text = uploaded_file.getvalue()
    # uploaded_file.close()
    # with open(uploaded_file, 'r') as file:
    #     input_text = file.read().replace('\n', '')

    return input_text

@st.cache(allow_output_mutation=True)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained('soroush/t5-finetuned-lesson-summarizer')
    tokenizer = T5Tokenizer.from_pretrained('soroush/t5-finetuned-lesson-summarizer')
    # model = T5ForConditionalGeneration.from_pretrained('./model/')
    # tokenizer = T5Tokenizer.from_pretrained('./model/')
    return tokenizer, model

def calculate_rouge(input_text, output_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(input_text, output_text)
    rouge1 = (100 * scores['rouge1'].fmeasure - 40.79) / 11.28 # normalize based on mean and std of nub summarizer on validation dataset
    rouge2 = (100 * scores['rouge2'].fmeasure - 17.76) / 10.09
    rougel = (100 * scores['rougeL'].fmeasure - 26.87) / 9.77
    return round(rouge1, 2), round(rouge2, 2), round(rougel, 2)

def run_summarizer(input_text, tokenizer, model):

    'brb...'

    latest_iteration = st.empty()
    bar = st.progress(5)
    # inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    # outputs = model.generate(
    #     inputs,
    #     max_length=150,
    #     length_penalty=1.0,
    #     num_beams=2,
    #     repetition_penalty=2.5,
    #     early_stopping=True)
    # output_text = tokenizer.decode(outputs[0])
    max_len = len(input_text)
    # try pipeline to avoid the truncation error
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    output = summarizer(input_text)
    output_text = output[0]['summary_text']
    if len(output_text) > max_len and max_len < 50:
        output_text = output_text[:max_len]

    for i in range(5, 100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Processing {i + 1}%')
        bar.progress(i + 1)
        time.sleep(0.01)
    latest_iteration.empty()

    bar.empty()
    st.markdown('**Summary:**')
    st.markdown('>'+output_text)
    rouge1, rouge2, rougel = calculate_rouge(input_text, output_text)
    st.write('**Summary Quality*: **')
    st.write('ROUGE-1: ', rouge1, 'ROUGE-2: ', rouge2, 'ROUGE-L: ', rougel)
    st.text('* F-1 measure normalized to mean and std on validation set')

    st.balloons()





if __name__==main():
    main()