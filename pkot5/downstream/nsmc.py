import os
from statistics import mean

import fire
import numpy as np
from datasets import load_dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainingArguments, Trainer, EvalPrediction


def train(model_name):
    dataset = load_dataset("nsmc")
    train_data = dataset['train']
    test_data = dataset['test']

    print(f"ex) {train_data[0]}")

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    dataset = []
    for data in [train_data, test_data]:
        all_input_ids = tokenizer([row['document'] for row in data], add_special_tokens=True, max_length=512, truncation=True).input_ids
        all_labels = tokenizer(['긍정적 댓글' if row['label'] == 1 else '부정적 댓글' for row in data], add_special_tokens=True).input_ids

        data = [
            dict(input_ids=input_ids, labels=labels)
            for input_ids, labels in zip(all_input_ids, all_labels)
        ]
        dataset.append(data)
    train_data, test_data = dataset

    args = Seq2SeqTrainingArguments(
        "pko-t5-nsmc",
        overwrite_output_dir=True,
        learning_rate=1e-3,
        optim='adafactor',
        warmup_ratio=0.6,
        num_train_epochs=5,
        local_rank=int(os.getenv("LOCAL_RANK", "-1")),

        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,

        evaluation_strategy='epoch',
        save_strategy='no',

        predict_with_generate=True,
        generation_max_length=5,
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def _compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_ids[label_ids < 0] = 0
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        acc = mean(1. if pred == label else 0. for pred, label in zip(predictions, labels))
        return {'accuracy': acc}

    Trainer(
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        compute_metrics=_compute_metrics
    ).train()


if __name__ == '__main__':
    fire.Fire(train)
