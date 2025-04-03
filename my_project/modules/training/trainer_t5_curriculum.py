#python -m modules.training.trainer_t5_curriculum
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from modules.training.dataloader_t5_jsonl import T5Dataset
from modules.training.eval.metrics_evaluator import Evaluator

def train_t5_with_curriculum(
    model_name="t5-small",
    dataset_path="data/train_shuffled_curriculum.jsonl",
    output_dir="output/t5_curriculum",
    eval_steps=500,
    max_epochs=3
):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    phases = [("easy", 1), ("medium", 1), ("all", max_epochs - 2)]
    for phase, epochs in phases:
        print(f"Starting phase: {phase}, epochs={epochs}")
        if phase == "all":
            train_dataset = T5Dataset(dataset_path, tokenizer)
        else:
            train_dataset = T5Dataset(dataset_path, tokenizer, difficulty=phase)

        args = TrainingArguments(
            output_dir=f"{output_dir}_{phase}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            num_train_epochs=epochs,
            save_total_limit=1,
            logging_dir=f"{output_dir}_{phase}/logs",
            learning_rate=5e-4,
            logging_steps=100,
            save_steps=500
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        model.save_pretrained(f"{output_dir}_{phase}/model")
        tokenizer.save_pretrained(f"{output_dir}_{phase}/model")
