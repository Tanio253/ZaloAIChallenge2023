import argparse
import json
import re
import jsonlines
from fraction import Fraction
from transformers import (
  AutoModelForCausalLM, 
  BitsAndBytesConfig, 
  GenerationConfig,
  AutoTokenizer)
from vllm import LLM, SamplingParams
import sys
import torch
MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

# def extract_answer_number(completion):
#     text = completion.split('The answer is: ')
#     if len(text) > 1:
#         extract_ans = text[-1].strip()
#         match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
#         if match:
#             if '/' in match.group():
#                 denominator = match.group().split('/')[1]
#                 numerator = match.group().split('/')[0]
#                 if is_number(denominator) == True and is_number(numerator) == True:
#                     if denominator == '0':
#                         return round(float(numerator.replace(',', '')))
#                     else:
#                         frac = Fraction(match.group().replace(',', ''))
#                         num_numerator = frac.numerator
#                         num_denominator = frac.denominator
#                         return round(float(num_numerator / num_denominator))
#                 else:
#                     return None
#             else:
#                 if float(match.group().replace(',', '')) == float('inf'):
#                     return None
#                 return round(float(match.group().replace(',', '')))
#         else:
#             return None
#     else:
#         return None

def extract_answer_number(completion):
    print(completion)
    text = completion.split('output')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'\d+\.\d+', extract_ans).group()
        return match

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=50, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    # llm = LLM(model='llm-agents/tora-code-7b-v1.0',tensor_parallel_size=tensor_parallel_size)
    quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                             bnb_4bit_compute_dtype = torch.bfloat16,
                                             bnb_4bit_quant_type = 'nf4',
                                             bnb_4bit_use_double_quant = True,)
    
    tokenizer = AutoTokenizer.from_pretrained('llm-agents/tora-code-7b-v1.0')
    tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig(top_p = 1, pad_token_id = tokenizer.eos_token_id, max_new_tokens = 512)                                        
    llm = AutoModelForCausalLM.from_pretrained('llm-agents/tora-code-7b-v1.0',
                                               quantization_config = quantization_config,
                                               device_map = 'auto',
                                               offload_folder = 'offload',
                                               offload_state_dict = True)
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        # if isinstance(prompt, list):
        #     pass
        # else:
        #     prompt = [prompt]

        # completions = llm.generate(prompt, sampling_params)
        # for output in completions:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     res_completions.append(generated_text)
        
        # print(prompt)
        # print(prompt_answer)

        inputs = tokenizer(prompt, return_tensors = 'pt').to('cuda')
        output = llm.generate(**inputs, generation_config = generation_config)
        generated_text = tokenizer.decode(output[0], skip_special_tokens = True)
        res_completions.append(generated_text)
        print('prompt: ', prompt)
        print(extract_answer_number(generated_text))

    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
            print(y_pred)
            print('----'*50)
            print(prompt_answer)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='/home/wliu/longhui/llms-all/gsm8k-inverse/data/test_use.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()
    # module load cuda/11.7
    # export TORCH_EXTENSIONS_DIR=./tmp
    # export PATH=/home/wliu/anaconda3/envs/llama_adapter/bin:$PATH
    # python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/lora_7b
    # python eval_math.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/lora_7b
    # python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/lora_7b_cosine
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/llama-70b-merged-qlora
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test( data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)

# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-re'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-back'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-merge'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_240k'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_merge_353k'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_merge_no_special_353k'
# MODEL_DIR='/lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_no_special_240k'

# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_no_special_240k
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_merge_no_special_353k
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_merge_353k
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-gsm_240k

# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-for
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-re
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-back
# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_llama-7b-merge



# python eval_wizard_gsm8k.py --model /lustre/fast/fast/wliu/longhui/inverse_ckpt/MATH_gsm_llama-7b-398k --tensor_parallel_size 8 --batch_size 400 --data_file /home/wliu/longhui/llms-all/gsm8k-inverse/data/test_use.jsonl