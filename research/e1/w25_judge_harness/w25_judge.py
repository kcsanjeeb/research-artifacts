#!/usr/bin/env python
"""W2.5 RVS judge harness.

Grades ReKV open-ended VQA predictions with a local LLM judge (vLLM offline)
using the EXACT prompt from the ReKV repo video_qa/eval/eval_open_ended.py
(Video-ChatGPT lineage judge used for RVS open-ended Accuracy + Score).

Metrics (as in eval_open_ended.py main):
  Accuracy = yes / (yes + no)
  Score    = mean of integer score 0..5

Usage:
  python w25_judge.py --pred_csv results.csv --out results.judged.json \
      --model /path/to/judge --tp 4 [--subset N] [--seed 2024]
"""
import argparse
import ast
import json
import re

import pandas as pd


def build_prompt(question, answer, pred):
    system = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    user = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_judge_output(text):
    """Return (pred in {yes,no,None}, score int or None)."""
    text = text.strip()
    try:
        d = ast.literal_eval(text)
        pred = str(d.get("pred", "")).lower()
        score = int(float(d.get("score")))
        return ("yes" if "yes" in pred else "no" if "no" in pred else None, score)
    except Exception:
        pass
    m = re.search(r"'pred':\s*'(\w+)'", text)
    pred = m.group(1).lower() if m else None
    m = re.search(r"'score':\s*([0-9.]+)", text)
    score = int(float(m.group(1))) if m else None
    return ("yes" if pred == "yes" else "no" if pred == "no" else None, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0, help="grade only first N rows (fixed prefix)")
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if args.subset:
        df = df.head(args.subset).copy()
    rows = df.to_dict(orient="records")

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len,
              enforce_eager=False)
    tok = llm.get_tokenizer()
    chat_prompts = [
        tok.apply_chat_template(build_prompt(r["question"], r["answer"], r["pred_answer"]),
                                tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    sp = SamplingParams(temperature=0, max_tokens=64)
    outs = llm.generate(chat_prompts, sp)

    yes = no = unscored = 0
    score_sum = 0
    judged = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        pred, score = parse_judge_output(text)
        if score is None:
            unscored += 1
        else:
            score_sum += score
        if pred == "yes":
            yes += 1
        elif pred == "no":
            no += 1
        judged.append({**r, "judge_raw": text, "judge_pred": pred, "judge_score": score})

    n_scored = yes + no
    accuracy = yes / n_scored if n_scored else 0.0
    n_scores = len(judged) - unscored
    avg_score = score_sum / n_scores if n_scores else 0.0
    result = {
        "judge_model": args.model,
        "n_questions": len(judged),
        "n_yes": yes, "n_no": no, "n_unparsed_pred": len(judged) - n_scored,
        "n_unparsed_score": unscored,
        "accuracy": accuracy,
        "average_score": avg_score,
        "judged": judged,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Accuracy: {accuracy*100:.1f}%  Score: {avg_score:.2f}  (n={len(judged)}, yes={yes}, no={no}, unparsed={len(judged)-n_scored})")


if __name__ == "__main__":
    main()
