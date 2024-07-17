"""
Example script to download any arbitrary model and format the repo correctly.
"""
import os
import gc
import math
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, Trainer, BitsAndBytesConfig
from utilities.utils import save_model
from huggingface_hub import login
import wandb
import torch
import dotenv
from scoring.common import EvaluateModelRequest
from scoring.entrypoint import _dl_dataset

dotenv.load_dotenv()

login(
    token=os.environ["HF_ACCESS_TOKEN"],
)


# model_name = "TapiwaWorknesh/m1"
model_name = "tensorwa/k703"
# model_name = "Sao10K/L3-8B-Stheno-v3.2"
save_path = "local_models"
chat_template_type = "chatml"

_dl_dataset()
wandb.init(entity='keisoft108', project='sn11')

# 1. Prepare dataset and collator
# 1-1. import packages
from transformers import DataCollatorForLanguageModeling
from scoring.common import chat_template_mappings
from scoring.dataset import PippaDataset
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from scoring.common import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    PROB_TOP_K,
    SAMPLE_SIZE,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EvaluateModelRequest,
    CREATIVITY_SCALE_FACTOR
)

from model.scores import (
    CREATIVITY_STEEPNESS,
    CREATIVITY_THRESHOLD
)
max_entropy = math.log(VOCAB_TRUNCATION)
max_len = MAX_SEQ_LEN

# 1-2. define dataset and set chat_template_params
dataset = PippaDataset(
    "datasets/pippa_deduped.jsonl",
    max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
input_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", force_download=True)
output_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", force_download=True)
if input_tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    input_tokenizer.pad_token = input_tokenizer.eos_token
    output_tokenizer.pad_token = output_tokenizer.eos_token
dataset.set_chat_template_params(chat_template_mappings[chat_template_type], input_tokenizer)

# 1-3. define data collator to set eos for vibe score training
class DataCollatorWithEOS(DataCollatorForLanguageModeling):
    def __call__(self, batch):
        batch = super().__call__(batch)
        
        # for i, input_ids in enumerate(batch['input_ids']):
        #     input_ids[max_len - MAX_GENERATION_LENGTH + batch['last_user_message_len'][i]] = input_tokenizer.eos_token_id
        return batch
data_collator = DataCollatorWithEOS(tokenizer=input_tokenizer, mlm=False)

# 1-4. Function to tokenize and encode the inputs and targets
def preprocess_function(examples):
    global log_once
    inputs = input_tokenizer(
        examples["contexts"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len - MAX_GENERATION_LENGTH,
        add_special_tokens=True,
    )  # this will put padding to the left and truncate the input if it is too long
    last_user_messages = input_tokenizer(
        examples["last_user_messages"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len - MAX_GENERATION_LENGTH,
        add_special_tokens=True,
    )
    targets = output_tokenizer(
        examples["target_texts"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_GENERATION_LENGTH,
        add_special_tokens=False,  # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
    )  # this will put padding to the right and truncate if necessary

    last_user_messages_len = torch.sum(last_user_messages["attention_mask"], dim=1)
    # for idx in range(0, BATCH_SIZE):

    # concatenate the inputs and targets and their attention masks using torch.cat
    input_ids = torch.cat((inputs["input_ids"], targets["input_ids"]), dim=1)
    attention_mask = torch.cat((inputs["attention_mask"], targets["attention_mask"]), dim=1)
    targets_ids_mask = torch.cat(
        [torch.zeros_like(inputs["attention_mask"]), targets["attention_mask"]],
        dim=1,
    )
    target_labels = input_ids.clone()
    target_labels[:, :inputs["input_ids"].shape[1]] = -100
    target_labels[attention_mask == 0] = -100
    # if last_user_messages_len[0] < MAX_GENERATION_LENGTH:
    #     target_labels[0, max_len - MAX_GENERATION_LENGTH + last_user_messages_len[0]] = output_tokenizer.eos_token_id
    

    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": target_labels[0]
    }

# 1-5. Preprocess the dataset
BATCH_SIZE = 2
tokenized_dataset = dataset.map(preprocess_function, BATCH_SIZE)
eval_tokenized_dataset = dataset.random_map(1024, preprocess_function, BATCH_SIZE)


# 2. Prepare Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    # eval_steps=1,
    logging_steps=1,
    logging_dir="./logs",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    lr_scheduler_kwargs = {
        "num_warmup_steps": 0,
        "num_training_steps": 26200
    },
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    num_train_epochs=100,
    weight_decay=0.01,
    gradient_accumulation_steps=32,
    report_to='wandb'
)

# 3. Custom callback to save model for each epoch and update eval dataset 
from scoring.eval_score import eval_score
# from scoring.vibe_score import calculate_vibe_match_score

# 4. Initialize Trainer
max_entropy = math.log(VOCAB_TRUNCATION)
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_score = 0
        self.best_creativity_score = 0

    def log(self, logs: dict) -> None:
        # Add custom logs
        custom_logs = {"eval_score": self.best_eval_score, "creativity_score": self.best_creativity_score}  # Example custom log
        logs.update(custom_logs)
        # Call the original log method
        super().log(logs)

    def adjusted_q_score(
            initial_score: float, creativity_score: float, threshold=CREATIVITY_THRESHOLD, steepness=CREATIVITY_STEEPNESS
        ):
            adjusted_score = initial_score / (1 + math.exp(-steepness * (creativity_score - threshold)))
            return adjusted_score

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # output = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        # print(f"Best/Current Eval Loss: {state.log_history[-1]['eval_loss']}/{self.best_loss}.\n")

        sampled_data = dataset.sample_dataset(SAMPLE_SIZE) 

        # vibe_contexts, vibe_target_texts, vibe_last_user_messages = zip(*sampled_data)
        # vibe_score = calculate_vibe_match_score(model, tokenizer, vibe_contexts, vibe_last_user_messages, vibe_target_texts)
        # print(f"Vibe score: {vibe_score}\n")
        
        eval_score_data = eval_score(
            model,
            sampled_data,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
        )
        evaluation_score = self.adjusted_q_score(eval_score_data.average_prob, eval_score_data.average_entropy)

        # print(f"Model evaluation score: {evaluation_score}\n")
        if self.best_eval_score < evaluation_score:
            self.best_eval_score = evaluation_score
            self.best_creativity_score = eval_score_data.average_entropy
            save_model(model, tokenizer, save_path, "best_model")
        else:
            save_model(model, tokenizer, save_path, "latest_model")
        
        self.eval_dataset = dataset.random_map(1024, preprocess_function, BATCH_SIZE)
        return {
            'eval_score': evaluation_score,
            'creativity_score': eval_score_data.average_entropy
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        # get the mask that only give us the output ids
        targets_ids_mask = inputs["attention_mask"].clone()
        targets_ids_mask[:, :max_len - MAX_GENERATION_LENGTH] = 0
        input_ids = inputs["input_ids"]

        # shift the output mask to the right by one to get the corresponding predicted logits
        targets_ids_mask = torch.cat(
            [torch.zeros_like(targets_ids_mask[:, :1]), targets_ids_mask[:, :-1]],
            dim=1,
        ).to("cuda")
    
        loss, outputs = super().compute_loss(model, inputs, return_outputs = True)

        # shift the logits to the right by one to get the corresponding predicted logits
        outputs.logits = torch.cat(
            [torch.zeros_like(outputs.logits[:, :1, :]), outputs.logits[:, :-1, :]],
            dim=1,
        )

        if torch.isnan(outputs.logits).any():
            raise ValueError("NaN values detected llm -> outputs.logits tensor")

        # Only keep the top PROB_TOP_K scores by -inf the rest
        # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized

        # get the top k logits and mask out the rest
        top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
        outputs.logits = torch.full_like(outputs.logits, float("-inf")).scatter(-1, top_k_indices, top_k_logits)


        # normalize the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)

        batch_entropy = (entropy * targets_ids_mask).sum() / targets_ids_mask.sum()
        normalized = batch_entropy.item() / max_entropy
        scaled_entropy = 1 - math.exp(-CREATIVITY_SCALE_FACTOR * normalized)

        if torch.isnan(probabilities).any():
            raise ValueError("NaN values detected in the probabilities tensor")

        # Get the top PROB_TOP_K indices and zero out all other probabilities
        top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
        mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
        probabilities[~mask] = 1e-9
        # Get the probabilities assigned by the model to the target tokens
        token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        # Mask out non target tokens
        token_probabilities = token_probabilities * targets_ids_mask

        # get the 1, 2, 3, 4 gram probabilities
        token_count = targets_ids_mask.sum().cpu().item()
        # 1-gram
        one_gram_probabilities = token_probabilities
        n_gram_prob = (one_gram_probabilities.sum().cpu().item() / token_count) * 0.25
        # 2-gram
        two_gram_probabilities = one_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-1]
        n_gram_prob += (two_gram_probabilities.sum().cpu().item() / token_count) * 0.25
        # 3-gram
        three_gram_probabilities = two_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-2]
        n_gram_prob += (three_gram_probabilities.sum().cpu().item() / token_count) * 0.25
        # 4-gram
        four_gram_probabilities = three_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-3]
        n_gram_prob += (four_gram_probabilities.sum().cpu().item() / token_count) * 0.25
        # custom_loss = (1 - n_gram_prob)*1.5 + loss * 0.5
        custom_loss = (1 - scaled_entropy) * 0.6 + loss * 0.2 + (1 - n_gram_prob)* 0.2

        # delete the tensors to free up memory
        del (
            targets_ids_mask,
            probabilities,
            token_probabilities,
            one_gram_probabilities,
            two_gram_probabilities,
            three_gram_probabilities,
        )
        del (
            four_gram_probabilities,
            n_gram_prob,
            mask,
            top_prob_indices,
            top_k_logits,
            top_k_indices,
            batch_entropy
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        return (custom_loss, outputs) if return_outputs else custom_loss

# 5. Load model
print(f"Loading model {model_name}")
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,  # This does not hurt performance much according to
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

from peft import LoraConfig, get_peft_model 
Lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)
# Lora_config = LoraConfig(
#     r=4,
#     lora_alpha=16,
#     lora_dropout=0,
#     target_modules=['q_proj', 'v_proj'],
#     bias="none",
#     task_type='CAUSAL_LM'
# )

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/dippy-bittensor-subnet/local_models/best_model",
    local_files_only=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # quantization_config=quant_config,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# lora_config = LoraConfig.from_pretrained(model_name)
# peft_model = get_peft_model(model, Lora_config)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_tokenized_dataset
    # data_collator=data_collator,
    # callbacks=[callback]
)

# 6. Train the model
trainer.train()
