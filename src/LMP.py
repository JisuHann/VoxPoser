
from time import sleep
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time
from LLM_cache import DiskCache
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoTokenizer
# 전역 모델과 프로세서 초기화
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
# stop_criteria = StoppingCriteriaList([
#     StopOnEosOrString(["<|endoftext|>", "# done"], processor)
# ])


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=True, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        self.eos_token='# done'

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, chat=True, **kwargs):
        """LLaVA 모델을 사용한 캐시된 API 호출"""
        prompt_txt = kwargs.pop('prompt', None)
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 128)
        
        if prompt_txt is None:
            raise ValueError("prompt_txt is required")
        
        try:
            # LLaVA 모델을 사용한 생성
            if chat:
                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt_txt}]}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)
            else:   
                inputs = processor(
                    text=prompt_txt,
                    return_tensors="pt"
                ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0)
                    # stopping_criteria=stop_criteria,
                    # eos_token_id=tokenizer.eos_token_id)
            
            response = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True,
            )
            return response
            
        except Exception as e:
            print(f"[LMP] API call failed: {e}")
            return ""

    def __call__(self, query, **kwargs):
        # prepare observation context
        print("NOW RUNNING LMP, ", self._name)
        if 'planner' in self._name:
            prompt_txt, user_query = self.build_prompt(query[0])
        else:
            prompt_txt, user_query = self.build_prompt(query)
        # call LMP
        start_time = time.time()
        print("PROMPT: ", prompt_txt)
        print("-"*30)
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt_txt,
                    temperature=self._cfg['temperature'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except Exception as e:
                print(f'Model generation error: {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** API call took {time.time() - start_time:.2f}s ***')
        
        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ').replace('</s>', '').replace('Query', '# Query').replace('\_','_')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])

            return_val = lvars[self._cfg['return_val_name']]()
            if return_val is None:
                breakpoint()
            return return_val


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        print("-----------Execute------------")
        print(code_str)
        import time, traceback

        # give generated code a readable filename to improve traceback
        filename = f"<generated:{int(time.time())}>"
        compiled = compile(code_str, filename, "exec")
        exec(compiled, custom_gvars, lvars)
    except Exception as e:
        print(f"[exec_safe] Exception type: {type(e).__name__}")
        print(f"[exec_safe] Exception message: {e}")

        # full traceback
        print("[exec_safe] Full traceback:")
        traceback.print_exception(type(e), e, e.__traceback__)

        # show the exact error line with a small context window
        try:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            err_lineno = tb.tb_lineno  # 1-based index within code_str
            code_lines = code_str.splitlines()
            start = max(0, err_lineno - 3)
            end = min(len(code_lines), err_lineno + 2)
            print(f"[exec_safe] Location: {filename}:{err_lineno}")
            for i in range(start, end):
                prefix = "=> " if (i + 1) == err_lineno else "   "
                print(f"{prefix}{i+1:4d}: {code_lines[i]}")
        except Exception as ctx_e:
            print(f"[exec_safe] Failed to render context: {ctx_e}")

        # small snapshot of available names can help spot missing symbols
        try:
            gkeys = sorted(list(custom_gvars.keys()))[:30]
            lkeys = sorted(list(lvars.keys()))[:30]
            print(f"[exec_safe] Globals (first 30): {gkeys}")
            print(f"[exec_safe] Locals (first 30): {lkeys}")
        except Exception:
            pass
        raise