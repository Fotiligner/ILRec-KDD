import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from trl import DPOTrainer, DPOConfig, ORPOTrainer, ORPOConfig
from trainer.ilrectrainer import ILRecTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoTokenizer, AutoModelForCausalLM

from Prompt import Prompt

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire
import json

random.seed(1958)
def train(
    #train
    output_dir="",
    logging_dir="",
    model_name ="",
    dataset="Instruments",
    gradient_accumulation_steps: int = 16,
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "Instruments-sft",
    wandb_name: str = "ILRec-Instruments-sft",   # the name of the wandb run
    # training hyperparameters
    beta: float = 0,   
    neg_num: int = 1,
    batch_size: int = 2,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    cutoff_len: int = 1024,
    eval_step = 0.05,  
):
    
    # os.environ['WANDB_PROJECT'] = wandb_project

    data_files = {
        "train": "./data/LC-Rec/Instruments_train_data.json",
        "validation": "./data/LC-Rec/Instruments_val_data.json"
    }


    index_item_file = "./data/LC-Rec/Instruments/Instruments.index.json"
    with open(index_item_file, "r") as file:
        dicting = json.load(file)

    item_search_table = {}
    for k, v in dicting.items():
        item_search_table["".join(v)] = int(k)

    def process_data(examples):
        dic = {"prompt":[], "chosen":[], "rejected":[], "chosen_id":[]}
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            dic['prompt'].append(examples['prompt'][i])
            dic['chosen'].append(examples['chosen'][i])
            dic['rejected'].append("")
            dic['chosen_id'].append(examples['chosen_id'][i])

        return dic

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(train_data)

    val_data = data["validation"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    is_adding = False

    if is_adding:
        new_tokens = set()
        with open("./data/LC-Rec/Instruments/Instruments.index.json", "r") as file:
            indices = json.load(file)

        for index in indices.values():
            for token in index:
                new_tokens.add(token)

        new_tokens = sorted(list(new_tokens))

        add_num = tokenizer.add_tokens(new_tokens)
        print("All new tokens added : ")
        print(add_num)


    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"
    config.use_cache=False
    load_type = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                torch_dtype=load_type,
                                                device_map=device_map)
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                #quantization_config=bnb_config) 这个是lora的时候要加回去的
    
    base_model.config.use_cache = False

    if is_adding:
        base_model.resize_token_embeddings(len(tokenizer))

    print("base_model_tokenizer_length")
    print(base_model.get_output_embeddings().weight.size(0))
    

    if resume_from_checkpoint != "":
        base_model = prepare_model_for_kbit_training(base_model)
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, 
                                        is_trainable=True)
    else:
        peft_config = LoraConfig(
        inference_mode=False,
        r=64,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        # bias="none",
        task_type="CAUSAL_LM",
        )

        base_model = get_peft_model(base_model, peft_config)

    base_model.print_trainable_parameters()

    # model_ref = AutoModelForCausalLM.from_pretrained(model_name,
    #                                             config=config,
    #                                             torch_dtype=load_type,
    #                                             device_map=device_map)
    #                                             # quantization_config=bnb_config)
    
    # if resume_from_checkpoint != "":
    #     reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    # else: 
    #     reference_model = model_ref

    training_args = ORPOConfig(
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
        beta=beta,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=100,
        evaluation_strategy="epoch",
        eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        # report_to = "wandb",
        # run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
        save_only_model=True,
        deepspeed="/home/hadoop-ba-dealrank/dolphinfs_hdd_hadoop-ba-dealrank/libingqian/BIGRec/LLMBox-bigrec/training/configs/ds_z3_bf16.json"
    )

    dpo_trainer = ILRecTrainer(
        base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        sasrec_input = None,
    )
    
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)