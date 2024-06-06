"""
Example script to download any arbitrary model and format the repo correctly.
"""
import os
import gc
import time
import random
import datetime as dt
import copy
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW, AutoConfig
from utilities.utils import save_model
from huggingface_hub import login
import dotenv
import math
import torch
import torch.nn as nn
import tqdm
import bittensor as bt
import huggingface_hub
from dippy_validation_api.dataset import PippaDataset
from dippy_validation_api.validation_api import chat_template_mappings

from bittensor.extrinsics.serving import get_metadata
import asyncio
from model.data import ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
import constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--compare_hotkeys",
    type=str,
    default=None,
    help="The hotkey of the model to check",
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-7,
    help="Learning rate",
)
parser.add_argument(
    "--merge_ratios",
    type=str,
    default=None,
    help="merge ratios joined by :",
)
parser.add_argument(
    "--load_from_repo",
    type=str,
    default=None,
    help="load from remote repo",
)
parser.add_argument(
    "--load_from_local",
    type=str,
    default=None,
    help="load from local dir",
)

bt.subtensor.add_args(parser)
args = parser.parse_args()
config = bt.config(parser)

subtensor = bt.subtensor(config=config)
subnet_uid = constants.SUBNET_UID
metagraph = subtensor.metagraph(subnet_uid)

wallet = None
model_metadata_store = ChainModelMetadataStore(subtensor, subnet_uid, wallet)

def get_model_from_key(hotkey):
    result = asyncio.run(model_metadata_store.retrieve_model_metadata(hotkey))
    model_name = f"{result.id.namespace}/{result.id.name}"
    print(model_name)
    return model_name


compare_models = []
model = None
tokenizer = None
model_name = None
if args.compare_hotkeys is not None:
    hotkeys = args.compare_hotkeys.split(":")
    for hotkey in hotkeys:
        model_name = hotkey
        if hotkey.startswith("5"):
            model_name = get_model_from_key(hotkey)
        print(f"Loading model for {model_name}/{hotkey}")
        model_i = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision=model_name.revision,
            # quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            # cache_dir=f"data/{str(request.hash)}",
            force_download=True
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        compare_models.append(model_i)

def add_pad_token(arg_tokenizer):
    new_pad_token = "<PAD>"
    num_added_tokens = arg_tokenizer.add_tokens(new_pad_token)
    # Update the tokenizer's padding token
    if num_added_tokens > 0:
        arg_tokenizer.pad_token = new_pad_token
    print(f"num added: {num_added_tokens}, {arg_tokenizer.pad_token_id}")

if config.load_from_repo is not None:
    model_name = config.load_from_repo
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, # This does not hurt performance much according to 
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # revision=model_name.revision,
        # quantization_config=quant_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # cache_dir=f"data/{str(request.hash)}",
        force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add_pad_token(tokenizer)
    # Function to add a new transformer block to the model
    # def add_transformer_block(model):
    #     new_layer = model.model.layers[0].__class__(model.config, len(model.model.layers))  # Create a new transformer block
    #     new_layer.to(torch.bfloat16)
    #     model.model.layers = torch.nn.ModuleList([*model.model.layers, new_layer])  # Ensure it is a ModuleList
    #     model.config.num_hidden_layers += 1
    
    # # Add the new layer
    # add_transformer_block(model)
    # # Reinitialize the new layer's parameters
    # for param in model.model.layers[-1].parameters():
    #     if param.dim() > 1:
    #         torch.nn.init.xavier_uniform_(param)
    #     else:
    #         torch.nn.init.zeros_(param)
    # add_transformer_block(model)
    # # Reinitialize the new layer's parameters
    # for param in model.model.layers[-1].parameters():
    #     if param.dim() > 1:
    #         torch.nn.init.xavier_uniform_(param)
    #     else:
    #         torch.nn.init.zeros_(param)    
    print(model)


elif config.load_from_local is not None:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.load_from_local,
        local_files_only=True,
        # revision=model_name.revision,
        # quantization_config=quant_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # cache_dir=f"data/{str(request.hash)}",
        # force_download=True
    )
    # model_config = model.config
    # model_config.transformers_version = "4.38.2"
    # model_config.save_pretrained("/workspace/dippy-bittensor-subnet/local-models/2024-05-29_18-08-16")
    model_name = "TapiwaWorknesh/git-kc11"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    if len(compare_models) > 0:
        model = copy.deepcopy(compare_models[0])
        if config.merge_ratios is not None and len(compare_models) > 1:
            merge_ratios = config.merge_ratios.split(":")
            merge_ratios = [float(r_str) for r_str in merge_ratios]
            tw = merge_ratios[0]
            for idx in range(1, len(compare_models)):
                ratio = tw/(tw+merge_ratios[idx])
                print(f"merging with ration {ratio}")
                model = merge(compare_models[idx], model, 1-ratio)
                tw += merge_ratios[idx]

# simple linear interpolation of weights between two models
def merge(model0, model1, ratio=0.5, embed_ratio=None, norm_ratio=None, fc_ratio=None): # higher ratio means more of model0
    if embed_ratio is None: embed_ratio = ratio # not sure if using different ratios for different parts of the model actually makes any sense
    if norm_ratio is None: norm_ratio = ratio
    if fc_ratio is None: fc_ratio = ratio

    params0 = {}
    for name, param in model0.named_parameters():
        params0[name] = param

    for name, param in model1.named_parameters():
        if "embed" in name:
            param.data = ((params0[name].data * embed_ratio) + (param.data * (1 - embed_ratio)))
        elif ("up_proj" not in name 
            and "down_proj" not in name 
            and "gate_proj" not in name 
            and "o_proj" not in name 
            and "k_proj" not in name 
            and "v_proj" not in name 
            and "q_proj" not in name
            and "embed" not in name
            ):
            param.data = ((params0[name].data * norm_ratio) + (param.data * (1 - norm_ratio)))
        elif "up_proj" in name or "down_proj" in name:
            param.data = ((params0[name].data * fc_ratio) + (param.data * (1 - fc_ratio)))
        else:
            param.data = ((params0[name].data * ratio) + (param.data * (1 - ratio)))

    return model1


MAX_GENERATION_LENGTH = 200
MAX_SEQ_LEN = 4096
SAMPLE_SIZE = 2080
VOCAB_TRUNCATION = 1000
PROB_TOP_K = 10

# create dir data if not exists
if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")

dataset = PippaDataset("data/pippa_deduped.jsonl", max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200)

chat_template_mappings = {
    "vicuna": "dippy_validation_api/prompt_templates/vicuna_prompt_template.jinja",
    "chatml": "dippy_validation_api/prompt_templates/chatml_prompt_template.jinja",
    "mistral": "dippy_validation_api/prompt_templates/mistral_prompt_template.jinja",
    "zephyr": "dippy_validation_api/prompt_templates/zephyr_prompt_template.jinja",
    "alpaca": "dippy_validation_api/prompt_templates/alpaca_prompt_template.jinja",
    "llama2": "dippy_validation_api/prompt_templates/llama2_prompt_template.jinja",
    "llama3": "dippy_validation_api/prompt_templates/llama3_prompt_template.jinja",
}


dotenv.load_dotenv()

login(
    token=os.environ["HF_ACCESS_TOKEN"],
)


# model_name = 'aks1s/aks-11-09'
save_path = 'local-models'
run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_dir_name = run_id

print(f"Loading model {model_name}\n")
# Load the tokenizer and model

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True, # This does not hurt performance much according to 
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # revision=model_name.revision,
#     # quantization_config=quant_config,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     device_map='auto',
#     # cache_dir=f"data/{str(request.hash)}",
#     force_download=True
# )
# print(f"Loaded model: {model}")
# base_model = copy.deepcopy(model)


# n_unfreeze_layers = 2
# freeze_params = []
# unfreeze_params = []
# if n_unfreeze_layers > 0:
#     for name, param in model.named_parameters():
#         parts = name.split(".")
#         if len(parts) <= 3:
#             # param.requires_grad = False
#             freeze_params.append(param)
#             continue
#         layer_idx = int(parts[2])
#         layer_count = 26
#         if layer_idx < layer_count - n_unfreeze_layers:
#             # param.requires_grad = False
#             freeze_params.append(param)
#         else:
#             unfreeze_params.append(param)

# # Define optimizer and learning rate scheduler
# optimizer_grouped_parameters = [
#     {"params": freeze_params, "lr": config.lr/10},
#     {"params": unfreeze_params, "lr": config.lr}
# ]

# optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
optimizer = AdamW(model.parameters(), lr=config.lr)
print(f"Start training with lr: {config.lr}")
num_epochs = 2000
batch_size = 5
accumulation_steps = 16
num_training_steps = num_epochs * MAX_SEQ_LEN / batch_size / accumulation_steps
num_warmup_steps = 0
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            
# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

print(f"model.config.max_position_embeddings: {model.config.max_position_embeddings}")
max_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)

# Training loop
# progress_bar = tqdm(range(num_training_steps))


bt.logging.on()

def calc_n_gram(outputs, targets_ids_mask, input_ids):
    # shift the logits to the right by one to get the corresponding predicted logits
    outputs.logits = torch.cat(
        [
            torch.zeros_like(outputs.logits[:, :1, :]), 
            outputs.logits[:, :-1, :]
        ], dim=1
    )

    if torch.isnan(outputs.logits).any():
        raise ValueError("NaN values detected llm -> outputs.logits tensor")

    # Only keep the top PROB_TOP_K scores by -inf the rest
    # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized

    # get the top k logits and mask out the rest
    top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
    outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)

    # normalize the logits to get probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()

    if torch.isnan(probabilities).any():
        raise ValueError("NaN values detected in the probabilities tensor")
    
    # Get the top PROB_TOP_K indices and zero out all other probabilities
    top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
    mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
    probabilities[~mask] = 1e-9
    # Get the probabilities assigned by the model to the target tokens
    token_probabilities = probabilities.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    # Mask out non target tokens
    token_probabilities = (1.1 - token_probabilities) * targets_ids_mask

    # get the 1, 2, 3, 4 gram probabilities
    token_count = targets_ids_mask.sum().cpu().item()
    # 1-gram
    one_gram_probabilities = token_probabilities
    n_gram_prob = (one_gram_probabilities.sum().cpu().item() / token_count - 0.1) * 0.01
    # 2-gram
    two_gram_probabilities = one_gram_probabilities[:, 1:] * token_probabilities[:, :-1]
    n_gram_prob += (two_gram_probabilities.sum().cpu().item() / token_count - 0.01) * 0.1
    # 3-gram
    three_gram_probabilities = two_gram_probabilities[:, 1:] * token_probabilities[:, :-2]
    n_gram_prob += (three_gram_probabilities.sum().cpu().item() / token_count - 0.001) * 0.5
    # 4-gram
    four_gram_probabilities = three_gram_probabilities[:, 1:] * token_probabilities[:, :-3]
    n_gram_prob += (four_gram_probabilities.sum().cpu().item() / token_count - 0.0001) * 2.5
    return n_gram_prob


def calc_prob(arg_model, arg_contexts, arg_target_texts, input_tokenizer, output_tokenizer):
    total_prob = 0
    n_count = 0
    arg_model.eval()
    with torch.no_grad():
        for i in range(0, len(arg_contexts), batch_size):
            random_batch = i
            targets = output_tokenizer(
                arg_target_texts[random_batch:random_batch+batch_size], 
                return_tensors='pt', 
                padding='max_length',
                truncation=True,
                max_length=MAX_GENERATION_LENGTH,
                add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
            )
            inputs = input_tokenizer(
                arg_contexts[random_batch:random_batch+batch_size], 
                return_tensors='pt', 
                padding='max_length',
                truncation=True, 
                max_length=max_len - MAX_GENERATION_LENGTH,
                add_special_tokens=True,
            ) # this will put padding to the left and truncate the input if it is too long
        
            # concatenate the inputs and targets and their attention masks using torch.cat
            input_ids = torch.cat((inputs['input_ids'], targets['input_ids']), dim=1).to('cuda')
            attention_mask = torch.cat((inputs['attention_mask'], targets['attention_mask']), dim=1).to('cuda')
        
            if input_ids.shape[1] > max_len:
                print(f"Input sequence length is greater than the maximum length the model can handle: {input_ids.shape[1]}")
                raise ValueError("Input sequence length is greater than the maximum length the model can handle")
        
            
            # get the mask that only give us the output ids
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(inputs['attention_mask']), 
                    targets['attention_mask']
                ], dim=1
            )
        
            # shift the output mask to the right by one to get the corresponding predicted logits
            targets_ids_mask = torch.cat(
                [
                    torch.zeros_like(targets_ids_mask[:, :1]), 
                    targets_ids_mask[:, :-1]
                ], dim=1
            ).to('cuda')
        
            # Get model predictions (logits)
            try:
                # print("Getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                outputs = arg_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=False, # don't use cache as we are not generating text. To prevent bug for Mistral models
                )
            except Exception as e:
                print("Error getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                raise ValueError("Error getting model predictions: " + str(e))
            
            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected in the logits tensor")
        
            # n_gram_prob = calc_n_gram(outputs, targets_ids_mask, input_ids)
            outputs.logits = torch.cat(
                [
                    torch.zeros_like(outputs.logits[:, :1, :]), 
                    outputs.logits[:, :-1, :]
                ], dim=1
            )
        
            if torch.isnan(outputs.logits).any():
                raise ValueError("NaN values detected llm -> outputs.logits tensor")
        
            # Only keep the top PROB_TOP_K scores by -inf the rest
            # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized
        
            # get the top k logits and mask out the rest
            top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
            outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)
        
            # normalize the logits to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()
        
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
            
            total_prob += n_gram_prob
            n_count += 1

            del outputs, targets_ids_mask, probabilities, token_probabilities, one_gram_probabilities, two_gram_probabilities, three_gram_probabilities
            del four_gram_probabilities, n_gram_prob, mask, top_prob_indices, top_k_logits, top_k_indices, inputs, targets

            gc.collect()
            torch.cuda.empty_cache()
    average_prob = total_prob / n_count
    return average_prob


input_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', force_download=True)
output_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', force_download=True)
# add_pad_token(input_tokenizer)
# add_pad_token(output_tokenizer)

print(f"IT pt : {tokenizer.pad_token}, {input_tokenizer.pad_token}/{input_tokenizer.pad_token_id}, Ot pt: {output_tokenizer.pad_token}/{output_tokenizer.pad_token_id}")
# if input_tokenizer.pad_token is None:
# input_tokenizer.pad_token = tokenizer.pad_token # add a pad token if not present
# input_tokenizer.pad_token_id = tokenizer.pad_token_id
# output_tokenizer.pad_token = tokenizer.pad_token # add a pad token if not present
# output_tokenizer.pad_token_id = tokenizer.pad_token_id

dataset.set_chat_template_params(chat_template_mappings['mistral'], input_tokenizer)
best_avg_loss = math.inf
best_prob = 0
validation_interval  = 1
max_validation_interval = 8
best_eval_prob = 0
total_eval_count = 0
eval_prob_sum = 0
patience_counter = 0
patience = 64
adjusted_lr = config.lr
save_on_eval = True
do_eval = False
random.seed(int(time.time()))
model.train()
last_sampled_data = None
for epoch in range(num_epochs):

    if save_on_eval and do_eval:
    # if save_on_eval:
        do_eval = False
        # sampled_data, last_sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
        sampled_data = dataset.sample_dataset_without(SAMPLE_SIZE, last_sampled_data)
        # sampled_data = dataset.sample_validation_data(100)
        contexts, target_texts, last_user_messages = zip(*sampled_data)
        # avg_compare_prob = []
        for compare_model in compare_models:
            avg_prob = calc_prob(compare_model, contexts, target_texts, input_tokenizer, output_tokenizer)
            print(f"Avg prob for model {avg_prob}")
            # avg_compare_prob.append(avg_prob)
        # assert compare_models[0] != model
        # avg_compare_prob.append(calc_prob(model, contexts, target_texts, input_tokenizer, output_tokenizer))
        # eval_prob = calc_prob(model, contexts, target_texts, input_tokenizer, output_tokenizer)
        total_prob = 0
        n_count = 0
        model_eval_path = "/workspace/dippy-bittensor-subnet/" + save_path + "/" + model_dir_name
        model_to_eval = None
        if epoch < 1:
            model_to_eval = model
        else:
            model_to_eval = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_eval_path,
                local_files_only=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        model_to_eval.eval()
        eval_batch_size = 4
        eval_log_lines = []
        with torch.no_grad():
            for i in range(0, len(contexts), eval_batch_size):
                targets = output_tokenizer(
                    target_texts[i:i+eval_batch_size], 
                    return_tensors='pt', 
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_GENERATION_LENGTH,
                    add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
                )
                inputs = input_tokenizer(
                    contexts [i:i+eval_batch_size], 
                    return_tensors='pt', 
                    padding='max_length',
                    truncation=True, 
                    max_length=max_len - MAX_GENERATION_LENGTH,
                    add_special_tokens=True,
                ) # this will put padding to the left and truncate the input if it is too long
            
                # concatenate the inputs and targets and their attention masks using torch.cat
                input_ids = torch.cat((inputs['input_ids'], targets['input_ids']), dim=1).to('cuda')
                attention_mask = torch.cat((inputs['attention_mask'], targets['attention_mask']), dim=1).to('cuda')
            
                if input_ids.shape[1] > max_len:
                    print(f"Input sequence length is greater than the maximum length the model can handle: {input_ids.shape[1]}")
                    raise ValueError("Input sequence length is greater than the maximum length the model can handle")
            
                
                # get the mask that only give us the output ids
                targets_ids_mask = torch.cat(
                    [
                        torch.zeros_like(inputs['attention_mask']), 
                        targets['attention_mask']
                    ], dim=1
                )
            
                # shift the output mask to the right by one to get the corresponding predicted logits
                targets_ids_mask = torch.cat(
                    [
                        torch.zeros_like(targets_ids_mask[:, :1]), 
                        targets_ids_mask[:, :-1]
                    ], dim=1
                ).to('cuda')
                # if total_eval_count % 2 == 1:
                #     targets_ids_mask[attention_mask == 0] = 0

            
                # Get model predictions (logits)
                try:
                    # print("Getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                    outputs = model_to_eval(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        use_cache=False, # don't use cache as we are not generating text. To prevent bug for Mistral models
                    )
                except Exception as e:
                    print("Error getting model predictions for sequence length: ", input_ids.shape[1], " batch size: ", input_ids.shape[0])
                    raise ValueError("Error getting model predictions: " + str(e))
                
                if torch.isnan(outputs.logits).any():
                    raise ValueError("NaN values detected in the logits tensor")
            
                # n_gram_prob = calc_n_gram(outputs, targets_ids_mask, input_ids)
                outputs.logits = torch.cat(
                    [
                        torch.zeros_like(outputs.logits[:, :1, :]), 
                        outputs.logits[:, :-1, :]
                    ], dim=1
                )
            
                if torch.isnan(outputs.logits).any():
                    raise ValueError("NaN values detected llm -> outputs.logits tensor")
            
                # Only keep the top PROB_TOP_K scores by -inf the rest
                # This will make the model only consider the top 100 tokens and make sure the models with higher vocab sizes are not penalized
            
                # get the top k logits and mask out the rest
                top_k_logits, top_k_indices = outputs.logits.topk(VOCAB_TRUNCATION, dim=-1)
                outputs.logits = torch.full_like(outputs.logits, float('-inf')).scatter(-1, top_k_indices, top_k_logits)
            
                # normalize the logits to get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cuda()
            
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

                if True:
                    # print the input tokens and top 10 predicted tokens
                    # eval_log_lines.append(f"Input: {input_tokenizer.decode(input_ids[0])}\n\n")
                    for j in range(len(input_ids[0])):
                        if targets_ids_mask[0][j].item() == 1:
                            actual_id = input_ids[0][j].item()
                            actual_token = output_tokenizer.decode([actual_id])
                            top_10_predicted_ids = outputs.logits[0][j].topk(10).indices.tolist()
                            top_10_predicted_tokens = [output_tokenizer.decode([id]) for id in top_10_predicted_ids]
                            eval_log_lines.append(f"Actual token: {actual_token} -> top 10 pred tokens: {top_10_predicted_tokens}, prob: {token_probabilities[0][j]}\n")
            
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

                # print(f"Target ids mask: {','.join(map(str, targets_ids_mask[0].tolist()))}")
                # print(f"Eval: {n_count} n_gram {n_gram_prob}", end='\r')
                eval_log_lines.append(f"Eval: {n_count}/{token_count} n_gram {n_gram_prob}/{one_gram_probabilities.sum().cpu().item()}/{two_gram_probabilities.sum().cpu().item()}/{three_gram_probabilities.sum().cpu().item()}/{four_gram_probabilities.sum().cpu().item()}\n\n\n")
                # pad_token_id = tokenizer.pad_token_id
                # real_inputs = input_ids[input_ids != pad_token_id]
                # time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # print(f"{time_str}     |     TRACK       |  - Step: {i} Input Ids: {real_inputs} ngram: {n_gram_prob}", end='\r')
                
                total_prob += n_gram_prob
                n_count += 1
    
                del outputs, targets_ids_mask, probabilities, token_probabilities, one_gram_probabilities, two_gram_probabilities, three_gram_probabilities
                del four_gram_probabilities, n_gram_prob, mask, top_prob_indices, top_k_logits, top_k_indices, inputs, targets
    
                gc.collect()
                torch.cuda.empty_cache()
        eval_prob = total_prob / n_count
        total_eval_count += 1
        eval_prob_sum += eval_prob
        avg_eval_prob = eval_prob_sum / total_eval_count
        eval_log_lines.append(f"Epoch {epoch + 1}/{num_epochs}, prob: {eval_prob} / {avg_eval_prob} / {best_eval_prob}\n")
        log_file_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = open(f"logs/{log_file_id}.txt", 'w')
        log_file.writelines(eval_log_lines)
        log_file.close()
        
        bt.logging.success(f"Epoch {epoch + 1}/{num_epochs}, prob: {eval_prob} / {avg_eval_prob} / {best_eval_prob}")
        if avg_eval_prob > best_eval_prob:
            best_eval_prob = avg_eval_prob
            save_model(model, tokenizer, save_path, "sn11_" + model_dir_name)
            validation_interval = max(validation_interval // 2, 1)
            patience_counter  = 0
            if adjusted_lr > config.lr:
                adjusted_lr = config.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = adjusted_lr
        else:
            validation_interval = min(validation_interval * 2, max_validation_interval)
            patience_counter += 1
        if patience_counter >= patience:
            adjusted_lr = adjusted_lr * 1.05
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr

    
    sampled_data, last_sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    contexts, target_texts, last_user_messages = zip(*sampled_data)
    
    epoch_loss = 0
    n_batches = 0
    n_acc_steps = 0
    total_prob = 0
    optimizer.zero_grad()
    loss_values = []
    prob_values = []
    for i in range(0, len(contexts), batch_size):
        targets = output_tokenizer(
            target_texts[i:i+batch_size], 
            return_tensors='pt', 
            padding='max_length',
            truncation=True,
            max_length=MAX_GENERATION_LENGTH,
            add_special_tokens=False # we don't want to add special tokens to the target as it continues from the context and already contains eos token.
        ) # this will put padding to the right and truncate if necessary

        inputs = input_tokenizer(
            contexts[i:i+batch_size], 
            return_tensors='pt', 
            padding='max_length',
            truncation=True, 
            max_length=max_len - MAX_GENERATION_LENGTH,
            add_special_tokens=True,
        ) # this will put padding to the left and truncate the input if it is too long
        
        last_um = input_tokenizer(
            last_user_messages[i:i+batch_size], 
            return_tensors='pt', 
            padding='max_length',
            truncation=True, 
            max_length=max_len - MAX_GENERATION_LENGTH,
            add_special_tokens=True,
        ) # this will put padding to the left and truncate the input if it is too long

        # concatenate the inputs and targets and their attention masks using torch.cat
        input_ids = torch.cat((inputs['input_ids'], targets['input_ids']), dim=1).to('cuda')
        attention_mask = torch.cat((inputs['attention_mask'], targets['attention_mask']), dim=1).to('cuda')

        if input_ids.shape[1] > max_len:
            print(f"Input sequence length is greater than the maximum length the model can handle: {input_ids.shape[1]}")
            raise ValueError("Input sequence length is greater than the maximum length the model can handle")

        
        # get the mask that only give us the output ids
        targets_ids_mask = torch.cat(
            [
                torch.zeros_like(inputs['attention_mask']), 
                targets['attention_mask']
            ], dim=1
        )
        # targets_ids_mask[input_ids == tokenizer.pad_token_id] = 0

        # # shift the output mask to the right by one to get the corresponding predicted logits
        targets_ids_mask = torch.cat(
            [
                torch.zeros_like(targets_ids_mask[:, :1]), 
                targets_ids_mask[:, :-1]
            ], dim=1
        ).to('cuda')
        # targets_ids_mask[attention_mask == 0] = 0

        # Shift the input ids to create labels
        pad_token_id = tokenizer.pad_token_id
        labels = input_ids.clone()
        labels[:, :inputs['input_ids'].shape[1]] = -100
        labels[attention_mask == 0] = -100
        
        outputs = model(
            input_ids=input_ids, 
            labels=labels,
            attention_mask=attention_mask,
            use_cache=False, # don't use cache as we are not generating text. To prevent bug for Mistral models
        )

        n_gram_prob = calc_n_gram(outputs, targets_ids_mask, input_ids)
        total_prob += n_gram_prob
        # diff = abs(last_um['attention_mask'].sum().cpu().item()/batch_size - targets['attention_mask'].sum().cpu().item()/batch_size) / MAX_GENERATION_LENGTH
        
        loss_track = outputs.loss.detach().item()
        # vive_weight = 0.001
        ngram_weight = 10
        loss_weight = 1
        loss = (outputs.loss * loss_weight + n_gram_prob * ngram_weight) / accumulation_steps
        loss_track_2 = loss.detach().item()
        loss.backward()

        if (n_batches + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if lr_scheduler.get_last_lr()[0] == 0.0:
                lr_scheduler.step()
            n_acc_steps += 1
            time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time_str}     |     TRACK       |  - Step: {n_acc_steps} loss/ngram: {loss_track}/{loss_track_2} / {n_gram_prob}", end='\r')
            loss_values.append(loss_track)
            prob_values.append(n_gram_prob)
        # progress_bar.update(1)
        epoch_loss += loss_track_2
        n_batches += 1
        
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate after each epoch
    # model.eval()
    # losses = []
    # for batch in eval_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #     loss = outputs.loss
    #     losses.append(loss.item())

    avg_loss = epoch_loss / n_batches
    avg_prob = total_prob / n_batches
    bt.logging.success(f"Epoch {epoch + 1}/{num_epochs}, Loss/Prob/Best: {avg_loss}/{best_avg_loss} / {avg_prob}/{best_prob}")

    # Check if the average loss of this epoch is the best we've seen so far
    if avg_loss < best_avg_loss:
        best_prob = avg_prob  # Update the best average loss
        best_avg_loss = avg_loss

        bt.logging.success(f"New best avg/prob: {best_avg_loss}/{best_prob}.")

        # Save the model to your mining dir.
        bt.logging.success(f"Saving model to path: {model_dir_name}.")
        save_model(model, tokenizer, save_path, model_dir_name)
        do_eval = True

