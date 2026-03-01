from openai import OpenAI
from time import sleep
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils.utils import load_prompt, DynamicObservation, IterableDynamicObservation, get_logger
import time
from utils.LLM_cache import DiskCache

logger = get_logger(__name__)

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench', llm_api_config=None):
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
        if llm_api_config is None:
            llm_api_config = {}
        self._client = OpenAI(
            base_url=llm_api_config.get('base_url', "http://172.17.0.1:8000/v1"),
            api_key=llm_api_config.get('api_key', "api-key-not-required"),
        )
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
    
    def _extract_code(self, msg):
        """Extract code from API response, handling reasoning model variants.

        - Standard models (Llama, Qwen3-2507): content only
        - Harmony channel models (gpt-oss-20b): vLLM puts final channel in content,
          analysis channel in reasoning_content
        - Think-tag models (e.g. QwQ, Qwen3 with thinking): </think> tag separates
          reasoning from final answer inside content
        """
        content = msg.content or ''
        reasoning = getattr(msg, 'reasoning_content', None) or ''

        # Strip <think>...</think> block if present in content
        if '</think>' in content:
            content = content.split('</think>', 1)[-1].strip()
            logger.debug('Stripped <think> block from content')

        # Use content (final answer) if non-empty, otherwise fall back to reasoning_content
        result = content if content.strip() else reasoning
        if not content.strip() and reasoning:
            logger.warning(
                f'[LMP "{self._name}"] content empty, falling back to reasoning_content '
                f'(model may not have reached final answer)'
            )

        # Clean up markdown code fences
        result = result.replace('```python', '').replace('```', '').strip()
        return result

    def _cached_api_call(self, **kwargs):
        user1 = kwargs.pop('prompt')
        new_query = '# Query:' + user1.split('# Query:')[-1]
        user1 = ''.join(user1.split('# Query:')[:-1]).strip()
        user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
        assistant1 = f'Got it. I will complete what you give me next.'
        user2 = new_query
        # handle given context (this was written originally for completion endpoint)
        if user1.split('\n')[-4].startswith('objects = ['):
            obj_context = user1.split('\n')[-4]
            # remove obj_context from user1
            user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
            # add obj_context to user2
            user2 = obj_context.strip() + '\n' + user2
        messages=[
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
        ]
        kwargs['messages'] = messages
        if kwargs in self._cache:
            logger.debug('(using cache)')
            return self._cache[kwargs]
        else:
            ret = self._client.chat.completions.create(**kwargs)
            msg = ret.choices[0].message
            ret = self._extract_code(msg)
            self._cache[kwargs] = ret
            return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except Exception as e:
                logger.warning(f'API error: {e} — retrying in 3s')
                sleep(3)
        logger.info(f'[LMP "{self._name}"] API call {time.time() - start_time:.2f}s')

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            logger.debug('#'*40 + f'\n## "{self._name}" generated code\n## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}')
        else:
            logger.debug('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                logger.error(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


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
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        logger.error(f'Error executing code:\n{code_str}')
        logger.error(f'Error message:\n{e}')
        raise e