import streamlit as st
import time
from transformers import AutoTokenizer, pipeline
from transformers.modeling_auto import AutoModelWithLMHead
# use this for heroku to avoid slug size issue
## torch==1.5.1
# torchvision==0.6.1

def main():
    st.title('Nub 1.0 Summarizer')
    input_text = st.text_area("Enter text")
    'or...'
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    tokenizer, model = load_model()
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

# @st.cache()
def load_model():
    # output_dir = '/Users/soroush/Documents/nub-summarizer/model'
    # model = T5ForConditionalGeneration.from_pretrained('soroush/model')
    # tokenizer = T5Tokenizer.from_pretrained('soroush/model')
    # https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    return tokenizer, model


def run_summarizer(input_text, tokenizer, model):

    'brb...'

    latest_iteration = st.empty()
    bar = st.progress(0)
    # inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    # outputs = model.generate(
    #     inputs,
    #     max_length=150,
    #     length_penalty=1.0,
    #     num_beams=2,
    #     repetition_penalty=2.5,
    #     early_stopping=True)
    # output_text = tokenizer.decode(outputs[0])

    # try pipeline to avoid the truncation error
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    output = summarizer(input_text)
    output_text = output[0]['summary_text']

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.01)


    '...and here\'s the summary'
    bar.empty()
    st.write(output_text)





if __name__==main():
    main()