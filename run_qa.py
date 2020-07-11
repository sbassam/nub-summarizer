from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
import argparse


def main(context_path, questions_path, output_dir):
    tokenizer = LongformerTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
    model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")



    def get_sent_id(input_id, max_startscore, max_endscore):
      sent_start_id = sent_end_id = 0
      all_tokens = tokenizer.convert_ids_to_tokens(input_id)
      for ind, token in enumerate(all_tokens):
        if (token == "." or token == "?") and ind < max_startscore:
          sent_start_id = ind + 1
        if (token == "." or token == "?") and ind > max_endscore:
          sent_end_id = ind
          break
      return sent_start_id, sent_end_id + 1

    summary_tuple = []
    context = "There are two broad categories of approaches to use for choosing stocks to buy or sell. They're based on Fundamental analysis and Technical analysis. Fundamental analysis involves looking at aspects of a company in order to estimate its value. Fundamental investors typically look for situations where the price of a company is below its value. Another camp is based on Technical analysis. Technicians don't care about the value of a company. Instead, they look for patterns or trends in a stock's price. This lesson will focus on technical analysis.  Let's start by looking at a few characteristics of technical analysis. First, what is it? One of the most important things to remember about technical analysis is that it looks only at price and volume. That's as opposed to fundamental analysis that looks at fundamental factors, like earnings, dividends, cash flow, book value, and so on. Technical analysis is price and volume only. We look back at historical price and volume to compute statistics on this time series, and these statistics are called indicators. Indicators are heuristics that may hint at a buy or sell opportunity. Now there is significant criticism of the technical approach. Folks consider that it's not an appropriate method for investing, because it's not considering the value of the companies. Instead, maybe you could think about it as a trading approach as opposed to an investing approach. "
    questions = ["How many categories of analysis are there?",
                 "What are the approaches based on?",
                 "What does fundamental analysis look at?",
                 "What are the two components of technical analysis?"
                 ]

    question_context_for_batch = []

    for question in questions:
        question_context_for_batch.append((question, context))

    encoding = tokenizer.batch_encode_plus(question_context_for_batch, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

    for index, (start_score, end_score, input_id) in enumerate(zip(start_scores, end_scores, input_ids)):
        max_startscore = torch.argmax(start_score)
        max_endscore = torch.argmax(end_score)
        sent_start_id, sent_end_id = get_sent_id(input_ids[index], max_startscore, max_endscore)
        ans_ids = input_ids[index][sent_start_id: sent_end_id]
        ans_tokens = tokenizer.convert_ids_to_tokens(ans_ids, skip_special_tokens=True)
        answer = tokenizer.convert_tokens_to_string(ans_tokens)
        # print ("\nQuestion: ",questions[index])
        print("Answer: ", answer)
        summary_tuple.append((sent_start_id, answer))
    summary_tuple = sorted(summary_tuple, key=lambda x: x[0])
    summary = ""
    summary_sent_ids = []
    for i in summary_tuple:
        if i[0] not in summary_sent_ids:
            summary_sent_ids.append(i[0])
            summary += i[1]
    print('SUMMARY: ', summary)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='automatic summary generation using question answering')
    parser.add_argument('--context_path', metavar='path', required=True,
                        help='path to context file')
    parser.add_argument('--questions_path', metavar='path', required=True,
                        help='path to questions')
    parser.add_argument('--output_dir', metavar='path', required=True,
                        help='where do you want to save the generated summaries?')
    args = parser.parse_args()
    main(context_path=args.context_path, questions_path=args.questions_path, output_dir=args.output_dir)