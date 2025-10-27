import torch
import os
from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria


# 전역 모델과 프로세서 초기화
model_name = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
if 'llava' in model_name:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=bnb_config, dtype=torch.bfloat16).to("cuda")
elif 'Qwen' in model_name:
    processor = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, dtype=torch.bfloat16).to("cuda")
else:
    raise ValueError(f"Invalid model name: {model_name}")
print("Successfully loaded model: ", model_name, " with 4 bit quantization")

# 2️⃣ 특정 문구 등장 시 멈추는 StoppingCriteria 정의
stop_words = "#done"
class StopOnKeywords(StoppingCriteria):
    def __init__(self, keywords, processor):
        self.keywords = keywords
        self.processor = processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        generated_text = self.processor.decode(input_ids[0], skip_special_tokens=True)
        for seq in self.keywords:
            if seq in generated_text:
                return True
        return False

# 3️⃣ 원하는 문구를 등록
# stop_criteria= StopOnKeywords(["done"], processor)
def refactor_output(output_txt):
    return output_txt.split("# done")[0]

def refactor_prompt(prompt_txt, system_prompt_path="/workspace/policy/Voxposer/prompts/robocasa/system_prompt.txt"):
    # set prompt
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = ""
    prompt_template = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_txt}]}
    ]
    return prompt_template