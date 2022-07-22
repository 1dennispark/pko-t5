import copy
import functools
import os
from pathlib import Path
from statistics import mean
import itertools

import fire
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainingArguments, Trainer, EvalPrediction, \
    TrainerCallback, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers.data.metrics import squad_metrics


@torch.no_grad()
def test(model_path: str):
    model_path = Path(model_path)

    dataset = load_dataset('squad_kor_v1')
    test_data = dataset['validation']

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(list(model_path.glob('checkpoint-*'))[0], use_fast=True)
    all_context_ids = tokenizer([r['context'] for r in test_data], add_special_tokens=False).input_ids
    all_prefix_ids = tokenizer([f"question: {r['question']} context: " for r in test_data], add_special_tokens=False).input_ids
    ori_test_data = test_data

    test_data, test_ids = [], []
    for row, context_ids, prefix_ids in zip(ori_test_data, all_context_ids, all_prefix_ids):
        id_ = row['id']
        context_len = 380
        all_ctx_ids = [context_ids[begin:begin+context_len] for begin in range(0, len(context_ids), 128)]
        all_ctx_texts = tokenizer.batch_decode(all_ctx_ids, skip_special_tokens=True)
        for ctx_ids, ctx in zip(all_ctx_ids, all_ctx_texts):
            test_data.append({
                'input_ids': prefix_ids + ctx_ids + [tokenizer.eos_token_id],
                'attention_mask': [1] * (len(prefix_ids) + len(ctx_ids) + 1),
            })
            test_ids.append(id_)

    for ckpt_dir in model_path.glob('checkpoint-*'):
        ckpt = int(ckpt_dir.name[len('checkpoint-')+1:])
        model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(ckpt_dir).cuda()
        eval_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

        all_logits, all_scores = [], []
        for data in tqdm(eval_dataloader, desc="evaluating.."):
            data = data.convert_to_tensors('pt').to(device='cuda')
            out = model.generate(input_ids=data['input_ids'],
                                 attention_mask=data['attention_mask'],
                                 max_length=16,
                                 num_beams=3,
                                 output_scores=True,
                                 return_dict_in_generate=True)

            all_logits += out.sequences.tolist()
            all_scores += out.sequences_scores.tolist()

        predictions = tokenizer.batch_decode(all_logits, skip_special_tokens=True)
        scores = all_scores

        pred_results = {}
        for pred, score, id_ in zip(predictions, scores, test_ids):
            pred_result = pred_results.get(id_, {'score': 0.0, 'answer': ''})
            if score > pred_result['score']:
                pred_result = {'score': score, 'answer': pred}
            pred_results[id_] = pred_result

        gold_results = {}
        for data in ori_test_data:
            gold_results[data['id']] = data['answers']['text'][0]

        assert len(list(pred_results.keys())) == len(list(gold_results.keys()))

        em, f1 = [], []
        for id_ in gold_results.keys():
            a_pred = pred_results[id_]['answer']
            a_gold = gold_results[id_]

            em.append(float(squad_metrics.compute_exact(a_gold, a_pred)))
            f1.append(float(squad_metrics.compute_f1(a_gold, a_pred)))

        em = 100. * sum(em) / len(em)
        f1 = 100. * sum(f1) / len(f1)

        print(f"ckpt-{ckpt} - em: {em:.2f} f1: {f1:.2f}")


def train(model_name):
    dataset = load_dataset("squad_kor_v1")
    train_data = dataset['train']

    print(f"ex) {train_data[0]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    answers = [row['answers']['text'][0] for row in train_data]
    all_context_ids = tokenizer([row['context'] for row in train_data], add_special_tokens=False).input_ids
    all_prefix_ids = tokenizer([f"question: {row['question']} context: " for row in train_data], add_special_tokens=False).input_ids
    all_label_ids = tokenizer(answers, add_special_tokens=True).input_ids

    train_data = []
    for context_ids, prefix_ids, label_ids, answer in tqdm(zip(all_context_ids, all_prefix_ids, all_label_ids, answers)):
        context_len = 380
        all_ctx_ids = [context_ids[begin:begin+context_len] for begin in range(0, len(context_ids), 128)]
        all_ctx_texts = tokenizer.batch_decode(all_ctx_ids, skip_special_tokens=True)
        for ctx_ids, ctx in zip(all_ctx_ids, all_ctx_texts):
            train_data.append({
                'input_ids': prefix_ids + ctx_ids + [tokenizer.eos_token_id],
                'attention_mask': [1] * (len(prefix_ids) + len(ctx_ids) + 1),
                'labels': [tokenizer.eos_token_id] if answer not in ctx else label_ids,
            })

    print(f"train_data length: {len(train_data)}")

    args = Seq2SeqTrainingArguments(
        "pko-t5-korquad",
        overwrite_output_dir=True,
        learning_rate=1e-3,
        optim='adafactor',
        warmup_ratio=0.6,
        num_train_epochs=2,
        local_rank=int(os.getenv("LOCAL_RANK", "-1")),

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,

        save_strategy='epoch',
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    )

    trainer.train()


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })
