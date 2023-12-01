import os
import sys
from typing import List
import json
from tqdm import tqdm
import wandb
import pandas as pd
import random
import numpy as np
import fire
import torch
import transformers
from transformers import GenerationConfig
from datasets import Dataset, load_metric

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYS_PREFIX = "<<SYS>>"
SYS_POSTFIX = " <</SYS>> "
INST_PREFIX = "<s> [INST] "
INST_POSTFIX = " "
OUTPUT_PREFIX = "[/INST] "
OUTPUT_POSTFIX = "</s>"


class Config:
  MODEL_ID = 'TheBloke/speechless-tora-code-7B-v1.0-GPTQ'
  REVISION = 'gptq-8bit-128g-actorder_True'
  OUTPUT_DIR = 'ToraZaloFT'
  PER_DEVICE_TRAIN_BATCH_SIZE = 4
  GRADIENT_ACCUMULATION_STEPS = 4
  OPTIM = 'paged_adamw_32bit' #8or32
  LEARNING_RATE = 2e-4
  LR_SCHEDULER_TYPE = 'constant'
  LOGGING_STEPS = 20
  SAVE_STRATEGY = 'steps'
  SAVE_STEPS = 20
  WARMUP_STEPS = 5
  EVAL_STEPS = 20
  LOGGING_DIR = './logs'
  MAX_STEPS = 240
  NUM_TRAIN_EPOCHS = 2
  FP16 = True
  PUSH_TO_HUB = False
  DATASET_TEXT_FIELD = 'content'
  MAX_SEQ_LENGTH = 4096
  REPORT_TO = 'wandb'
  PACKING = False
  DO_EVAL = True
  NEFTUNE_NOISE_ALPHA = 5
  EVALUATION_STRATEGY = 'steps'
  R = 128
  LORA_ALPHA = 256
  LORA_DROPOUT = 0.05
  TARGET_MODULES = ['q_proj', 'v_proj']
  BIAS = 'none'
  TASK_TYPE = 'CAUSAL_LM'
  
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess(data_point, tokenizer, cutoff_len):
    dialog = data_point['dialog']

    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    assert roles[0].upper() != "ASSISTANT"
    assert roles[-1].upper() == "ASSISTANT"

    input_messages = []
    if roles[0].upper() == "SYSTEM":
        input_messages.append(SYS_PREFIX+messages[0]+SYS_POSTFIX)

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            input_messages.append(msg + " " + OUTPUT_POSTFIX)
        elif role.upper() == "USER":
            input_messages.append(INST_PREFIX + msg + INST_POSTFIX + OUTPUT_PREFIX)

    tokenized_input = tokenizer(input_messages, add_special_tokens=False)

    input_ids = []
    labels = []

    if roles[0].upper() == "SYSTEM":
        input_ids.extend(tokenized_input.input_ids[0])
        labels.extend([-100]*len(tokenized_input.input_ids[0]))

    for role, msg in zip(roles, tokenized_input.input_ids):

        if role.upper() == "USER":
            labels.extend([-100]*len(msg))
            input_ids.extend(msg)
        
        elif role.upper() == "ASSISTANT":
            labels.extend(msg)
            input_ids.extend(msg)


    input_ids = torch.LongTensor(input_ids)[:cutoff_len]
    labels = torch.LongTensor(labels)[:cutoff_len]

    assert input_ids.shape == labels.shape

    return {
        "input_ids": input_ids,
        "labels": labels
    }

# def transformer_to_dialog(math_data):
#     dialogs = []

#     for d in math_data:
#         question = d['question']
#         choices = d['choices']
#         choices = "\n".join(choices)
#         answer = d['answer']
#         explanation = d['explanation']
#         dialog = [
#             {"role": "system", "content": "Bạn là một trợ lý giải toán. Hãy suy nghĩ từng bước một, \
# sau đó chọn 1 trong những đáp án A, B, C, D dưới đây. Điều này rất quan trọng với việc học của tôi."}
#         ]
#         dialog += [
#         {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn:\n{choices}.\nViết lời giải của bạn, sau đó đưa ra lựa chọn."},
#         {"role": "assistant", "content": f"Lời giải: {explanation}\nKết quả: {answer}"}
#         ]

#         dialogs.append(dialog)
        
#     return dialogs

# def transformer_for_test(data):
#     dialogs = []
#     for d in data:
#         question = d['question']
#         choices = d['choices']
#         choices = "\n".join(choices)
#         dialog = [
#             {"role": "system", "content": "Bạn là một trợ lý giải toán. Hãy suy nghĩ từng bước một, \
# sau đó chọn 1 trong những đáp án A, B, C, D dưới đây. Điều này rất quan trọng với việc học của tôi."},
#             {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn:\n{choices}.\nViết lời giải của bạn, sau đó đưa ra lựa chọn."},
#         ]
#         dialogs.append(dialog)

#     return dialogs

def get_dialog_string(dialog):
    prompt = ""
    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    if roles[0].upper() == "SYSTEM":
        prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            prompt += f" {msg} {OUTPUT_POSTFIX}"
        elif role.upper() == "USER":
            prompt += f" {INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"

    return prompt

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "mTrain.json",
    val_data_path: str = 'mVal.json',
    test_path: str = "math_test.json",
    output_dir: str = "./lora-zalo",
    # training hyperparams
    batch_size: int = 4,
    eval_batch_size: int = 4,
    micro_batch_size: int = 4,
    # num_epochs: int = 3,
    max_steps = 200,
    learning_rate: float = 2e-4,
    cutoff_len: int = 256,
    val_set_size: float = 0.3,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    optim: str = "paged_adamw_32bit",
    # lora hyperparams
    train_qlora: bool = False,
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        'o_proj', 
        'k_proj', 
        'down_proj', 
        'gate_proj', 
        'up_proj', 
        'lm_head'
    ],
    # llm hyperparams
    seed: int = 42,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    wandb_api_key: str = 'db6f204a887bc4436623a37ce6db19522e0069d5', # Wandb api key
    huggingface_token: str = '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', # token to login huggingface
    huggingface_repo: str = None, # push to repo
):
    
    gradient_accumulation_steps = 8
    seed_everything(seed)

    device_map = "auto"
    world_size = torch.cuda.device_count()
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if wandb_api_key is not None:
        use_wandb = True
        wandb.login(key=wandb_api_key)
    else:
        os.environ["WANDB_DISABLED"] = "true"
        use_wandb = False

    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code =True)

    tokenizer.pad_token = tokenizer.eos_token
    global OUTPUT_POSTFIX
    OUTPUT_POSTFIX = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Allow batched inference

    if train_qlora is True:
        optim="paged_adamw_8bit"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=bnb_config,
                use_safetensors=True
            )
        model = prepare_model_for_kbit_training(model)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            # torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code = True
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    # train_data = read_json(train_data_path)['data']
    # random.shuffle(train_data)
    # val_data = read_json(val_data_path)['data']
    # random.shuffle(val_data)

    # if val_set_size > 1:
    #     val_set_size = 0.3
    # val_set_size = int(val_set_size * len(data))
    # train_data = data[val_set_size:]
    # val_data = data[:val_set_size]

    countA = 0
    countB = 0
    countC = 0
    countD = 0
    with open(train_data_path, 'r') as file:
        train_data = json.load(file)
    with open(val_data_path, 'r') as file:
        val_data = json.load(file)
    train_data = train_data['data']
    len_train_data = len(train_data)
    print(f"Length train dataset: {len_train_data}")
    val_data = val_data['data']
    len_val_data = len(val_data)
    print(f"Length validation dataset: {len_val_data}")
    zalo_train_data = {'question': [],
                    'choices': [],
                    'answer': [],
                    'explanation': [],
                    'id': []}
    zalo_val_data = {'question': [],
                    'choices': [],
                    'answer': [],
                    'explanation': [],
                    'id': []}
    for i in range(len_train_data):
        zalo_train_data['question'].append(train_data[i]['question'])
        zalo_train_data['choices'].append(train_data[i]['choices'])
        zalo_train_data['answer'].append(train_data[i]['answer'])
        zalo_train_data['id'].append(train_data[i]['id'])
        zalo_train_data['explanation'].append(train_data[i]['explanation'])
        if 'A.' in train_data[i]['answer']:
            countA+=1
        elif 'B.' in train_data[i]['answer']:
            countB+=1
        elif 'C.' in train_data[i]['answer']:
            countC+=1
        elif 'D.' in train_data[i]['answer']:
            countD+=1
    print(f"Training Data:\nA: {countA} B: {countB} C: {countC} D: {countD}")
    countA = 0
    countB = 0
    countC = 0
    countD = 0
    for i in range(len_val_data):
        zalo_val_data['question'].append(val_data[i]['question'])
        zalo_val_data['choices'].append(val_data[i]['choices'])
        zalo_val_data['answer'].append(val_data[i]['answer'])
        zalo_val_data['id'].append(val_data[i]['id'])
        zalo_val_data['explanation'].append(val_data[i]['explanation'])
        if 'A.' in val_data[i]['answer']:
            countA+=1
        elif 'B.' in val_data[i]['answer']:
            countB+=1
        elif 'C.' in val_data[i]['answer']:
            countC+=1
        elif 'D.' in val_data[i]['answer']:
            countD+=1
    print(f"Validation Data:\nA: {countA} B: {countB} C: {countC} D: {countD}")
    
    system_prompt = """\
    Bạn là một trợ lý giải toán. Hãy suy nghĩ từng bước một, \
    sau đó chọn 1 trong những đáp án A, B, C, D dưới đây. Điều này rất quan trọng với việc học của tôi."""
    def preprocess(samples):
        prefix_prompt =  f"{SYS_PREFIX}{system_prompt}{SYS_POSTFIX}{INST_PREFIX}"
        choices_list = samples['choices'].copy()
        choices = "\n".join(choices_list)
        question = f"Câu hỏi: {samples['question']}\nLựa chọn:\n{choices}.\nViết lời giải của bạn, sau đó đưa ra lựa chọn."
        explanation = f"{samples['explanation']}"
        answer = f"Lời giải: {explanation}\nKết quả: {samples['answer']}"
        formatted_conv = f"{prefix_prompt}{question}{INST_POSTFIX}{OUTPUT_PREFIX}{answer} {OUTPUT_POSTFIX}"
        return {'content':formatted_conv}
    train_data = Dataset.from_dict(zalo_train_data, split = 'train')
    train_data = train_data.map(
        preprocess,
        # batched = True,
        remove_columns = train_data.column_names
    )
    train_data = train_data.shuffle(100)
    print(train_data[2])

    val_data = Dataset.from_dict(zalo_val_data, split = 'val')
    val_data = val_data.map(
        preprocess,
        # batched = True,
        remove_columns = val_data.column_names
    )
    val_data = val_data.shuffle(100)
    print(val_data[0])
    # print(count1)

    # train_ds = (
    #     Dataset.from_dict({"dialog": train_dialogs}, split = 'train').shuffle())
    # train_ds = train_ds.map(lambda x: preprocess(x, tokenizer, cutoff_len), remove_columns = train_ds.column_names)
    # train_ds = train_ds.filter(lambda x: len(x['input_ids']) < cutoff_len)

    # val_ds = (
    #     Dataset.from_dict({"dialog": val_dialogs}).map(lambda x: preprocess(x, tokenizer, cutoff_len))
    # ).filter(lambda x: len(x['input_ids']) < cutoff_len)


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    perplexity = load_metric("perplexity")
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=1)
        perplexity_val = perplexity.compute(predictions=predictions, references=labels)
        return {"perplexity": perplexity_val}

    training_arguments = TrainingArguments(
        output_dir = Config.OUTPUT_DIR,
        per_device_train_batch_size = Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS,
        optim = Config.OPTIM,
        learning_rate = Config.LEARNING_RATE,
        save_strategy = Config.SAVE_STRATEGY,
        lr_scheduler_type = Config.LR_SCHEDULER_TYPE,
        eval_steps = Config.EVAL_STEPS,
        logging_steps = Config.LOGGING_STEPS,
        max_steps = Config.MAX_STEPS,
        fp16 = Config.FP16,
        save_steps = Config.SAVE_STEPS,
        logging_dir = Config.LOGGING_DIR,
        report_to = Config.REPORT_TO,
        do_eval = Config.DO_EVAL,
        warmup_steps = Config.WARMUP_STEPS,
        push_to_hub = Config.PUSH_TO_HUB,
        neftune_noise_alpha = Config.NEFTUNE_NOISE_ALPHA,
        evaluation_strategy = Config.EVALUATION_STRATEGY,
        num_train_epochs = Config.NUM_TRAIN_EPOCHS
    )
    trainer = SFTTrainer(model = model,
                     args = training_arguments,
                     train_dataset = train_data,
                     eval_dataset = val_data,
                     peft_config = peft_config,
                     tokenizer = tokenizer,
                     packing = False,
                     dataset_text_field = 'content',
                     max_seq_length = Config.MAX_SEQ_LENGTH)
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)
    try:
        if isinstance(huggingface_token, str) and isinstance(huggingface_repo,str):
            from huggingface_hub import login
            login(token = huggingface_token)
            model.push_to_hub(
                huggingface_repo
            )
    except:
        pass
        
    # Start to Evaluate Data
#     model.eval()

#     # eval_rows = ValidateFunc(model=model,
#     #                          tokenizer=tokenizer,
#     #                          test_data=val_data,
#     #                          batch_size=eval_batch_size)
#     # cnt = 0
#     # total = len(eval_rows)
#     # for eval_row, eval_item in zip(eval_rows, val_data):
#     #     cnt += eval_row['answer'] == eval_item['answer']

#     # print(f"Eval Accuracy: {100 * cnt / total:.2f}")
#     # Test data
#     test_rows = ValidateFunc(model=model,
#                              tokenizer=tokenizer,
#                              test_path=test_path,
#                              batch_size=eval_batch_size)

#     df = pd.DataFrame(test_rows)
#     df.to_csv("zalo_submission.csv", index=False)


# def read_json(path):
#     f = open(path, encoding = "utf8")
#     data = json.load(f)
#     f.close()
#     return data

# def write_json(path, obj):
#     if not path.endswith(".json"):
#         path += ".json"

#     json_object = json.dumps(obj, indent=4, ensure_ascii=False)
#     with open(path, "w", encoding="utf-8") as outfile:
#         outfile.write(json_object)


# def generate_response(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
#     encoding = tokenizer(prompt, padding=True, 
#                          truncation=True, 
#                          return_tensors="pt", 
#                          max_length = max_length, 
#                          add_special_tokens=False)
#     input_ids = encoding["input_ids"].to(model.device)
#     attention_mask = encoding['attention_mask'].to(model.device)

#     generation_config = GenerationConfig(
#         temperature=temperature,
#         top_p=1,
#         do_sample = True,
#         num_beams = 1,
#         top_k = top_k,
#         pad_token_id = tokenizer.pad_token_id,
#         eos_token_id = tokenizer.eos_token_id
#     )

#     with torch.inference_mode():
#         return model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             max_new_tokens=512,
#         )

# def format_response(response, tokenizer):
#     if response.sequences.size(0) == 1:
#         decoded_output = tokenizer.decode(response.sequences[0], skip_special_tokens = True)
#         response = [decoded_output.split(OUTPUT_PREFIX)[-1].strip()]
#         # put to list to make it compatible
#     else:
#         decoded_outputs = tokenizer.batch_decode(response.sequences, skip_special_tokens=True)
#         response = []
#         for o in decoded_outputs:
#             response.append(o.split(OUTPUT_PREFIX)[-1].strip())
#     return response

# def ask_alpaca(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
#     response = generate_response(prompt, 
#                                  model, 
#                                  tokenizer, 
#                                  max_length = max_length,
#                                  temperature =temperature, 
#                                  top_k = top_k)
#     response = format_response(response, tokenizer)
#     return response

# def batch_inference(data, model, tokenizer, batch_size = 4, max_length = 1500, temperature = 0.1, top_k = 50):
#     tk = tqdm(range(0, len(data), batch_size))
#     predictions = []
#     for start_idx in tk:
#         batch = data[start_idx:start_idx+batch_size]
#         preds = ask_alpaca(batch, model, tokenizer, max_length = max_length, temperature =temperature, top_k = top_k)
#         predictions += preds
#         examples = [p[:50] for p in preds]
#         tk.set_postfix(
#             examples=examples,
#         )
#     return predictions

# def get_results(test_data, test_dialogs):
#     rows = []
#     for data, dialog in zip(test_data, test_dialogs):
#         id = data['id']
#         choices = data['choices']
#         answer = None
#         solution_return = dialog[-1]['content']
#         for idx, d in enumerate([('A.', '(A)', 'A:'), ('B.', '(B)', 'B:'), ('C.', '(C)', 'C:'), ('D.', '(D)', 'D:')]):
#             if any(i in solution_return for i in d):
#                 answer = choices[idx]

#         if answer is None:
#             rows.append({"id": id, "answer": choices[0]}) # if can't find
#             print(id, solution_return)
#         else:
#             rows.append({"id": id, "answer": answer})

#     return rows

# def ValidateFunc(model, tokenizer, test_path = None, test_data = None, batch_size = 8):
#     if test_data is None and test_path is not None:
#         test_data = read_json(test_path)['data']

#     test_dialogs = transformer_for_test(test_data)
#     prompts = [get_dialog_string(d) for d in test_dialogs]
#     responses = batch_inference(prompts, model, tokenizer, batch_size=1)

#     for dialog, response in zip(test_dialogs, responses):
#         dialog.append({
#             "role": "assistant",
#             "content": response
#         })

#     rows = get_results(test_data, test_dialogs)
#     return rows

if __name__ == "__main__":
    fire.Fire(train)
