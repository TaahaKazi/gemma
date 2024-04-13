import json
import os

os.environ['HF_TOKEN'] = json.loads(open('key.json').read())['HF_TOKEN']

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
from peft import LoraConfig
import transformers
from trl import SFTTrainer
from datasets import load_dataset


def formatting_func(example):
    output_texts = []
    for i in range(len(example['sql_prompt'])):
        text = f"### Input: {example['sql_prompt'][i]}\n ### Output: {example['sql'][i]}"
        output_texts.append(text)
    return output_texts


def run(model_id="google/gemma-7b"):
    model_id = "google/gemma-7b"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0},
                                                 token=os.environ['HF_TOKEN'])

    # Save the model
    os.environ["WANDB_DISABLED"] = "true"

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    data = load_dataset("gretelai/synthetic_text_to_sql")
    # TODO: Add data collator


    trainer = SFTTrainer(
        model=model,
        train_dataset=data["test"],
        max_seq_length=512,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=5,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()

    # Sample Test
    example = data["train"][0]
    text = f"Context: {example['sql_context']}\nTask: {example['sql']}\nSolution:\nExplanation: "
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Save the model
    model.save_pretrained("outputs")
    tokenizer.save_pretrained("outputs")

if __name__ == "__main__":
    run()