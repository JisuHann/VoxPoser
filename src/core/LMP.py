import ast
import bdb
import base64
import io
import os
import textwrap
from openai import OpenAI
from time import sleep
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from PIL import Image
from utils.utils import load_prompt, DynamicObservation, IterableDynamicObservation, get_logger
from utils.errors import LMPApiUnreachable, LMPEmptyOutput
from utils.LLM_cache import DiskCache
import time

logger = get_logger(__name__)

# Inner cap on per-call API retries. 5 * 3s = 15s — enough for a transient
# vLLM stall to recover, short enough that a dead endpoint surfaces to the
# outer loop quickly instead of hanging.
API_MAX_RETRIES = 5

# Module-level cache singleton — shared across all LMP instances in this
# process. Configured by configure_cache() (called from run_LMP startup) or
# defaults to enabled with cache_dir='cache'. Env vars override config:
#   LMP_DISABLE_CACHE=1   forces disable
#   LMP_CACHE_DIR=path    overrides cache directory
_GLOBAL_CACHE = None
_CACHE_ENABLED = True
_CACHE_DIR = 'cache'

def configure_cache(enabled=True, cache_dir='cache'):
    """Set cache config before first use. No-op if cache already initialized."""
    global _CACHE_ENABLED, _CACHE_DIR
    _CACHE_ENABLED = enabled
    _CACHE_DIR = cache_dir

def _get_cache():
    global _GLOBAL_CACHE
    # Env-var overrides
    if os.environ.get('LMP_DISABLE_CACHE', '0') == '1' or not _CACHE_ENABLED:
        return None
    if _GLOBAL_CACHE is None:
        cache_dir = os.environ.get('LMP_CACHE_DIR', _CACHE_DIR)
        _GLOBAL_CACHE = DiskCache(cache_dir=cache_dir, load_cache=True)
        logger.info(f'[LMP cache] enabled at {cache_dir} ({len(_GLOBAL_CACHE.data)} entries loaded)')
    return _GLOBAL_CACHE


class _CachedMsg:
    """Lightweight stand-in for openai ChatCompletion message for _extract_code."""
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content

import numpy as _np

class DeferredMap:
    """Wraps a callable map so LLM code can combine maps with numpy ops (e.g. map_a + map_b)."""
    def __init__(self, fn):
        self._fn = fn
    def __call__(self):
        return self._fn()
    def __add__(self, other):
        if isinstance(other, DeferredMap):
            return DeferredMap(lambda: _np.clip(self() + other(), 0, 1))
        return DeferredMap(lambda: _np.clip(self() + other, 0, 1))
    def __radd__(self, other):
        return self.__add__(other)
    def __mul__(self, other):
        if isinstance(other, DeferredMap):
            return DeferredMap(lambda: self() * other())
        return DeferredMap(lambda: self() * other)

_VLM_PATTERNS = ['vl', 'vision', 'pixtral', 'llava', 'internvl', '4.6v', 'cosmos', 'mimo']

def is_vlm(model_name):
    """Check if a model name indicates a Vision-Language Model."""
    name = model_name.lower()
    return any(p in name for p in _VLM_PATTERNS)

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench', llm_api_config=None):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._env = env
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._images = None
        self._image_labels = None
        if llm_api_config is None:
            llm_api_config = {}
        self._client = OpenAI(
            base_url=llm_api_config.get('base_url', "http://localhost:8000/v1"),
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

        # Cosmos-Reason2: extract code from <answer>...</answer> block.
        # The model card requires wrapping the answer in this tag pair.
        if '<answer>' in content and '</answer>' in content:
            ans_start = content.find('<answer>') + len('<answer>')
            ans_end = content.find('</answer>', ans_start)
            content = content[ans_start:ans_end].strip()
            logger.debug('Extracted code from <answer> block')

        # Use content (final answer) if non-empty, otherwise fall back to reasoning_content
        result = content if content.strip() else reasoning
        if not content.strip() and reasoning:
            logger.warning(
                f'[LMP "{self._name}"] content empty, falling back to reasoning_content '
                f'(model may not have reached final answer)'
            )

        # Clean up markdown code fences and model-specific wrapper tokens
        result = result.replace('```python', '').replace('```', '')
        result = result.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        # Some models output literal \n instead of actual newlines
        result = result.replace('\\n', '\n').strip()

        # Strip import lines (exec_safe bans them; fixed_vars already provides imports)
        lines = result.split('\n')
        import_stripped = [l for l in lines if not l.strip().startswith(('import ', 'from '))]
        if len(import_stripped) < len(lines):
            logger.debug(f'[LMP "{self._name}"] stripped {len(lines) - len(import_stripped)} import line(s)')
            result = '\n'.join(import_stripped).strip()

        # Validate as Python; if invalid, filter out natural-language lines
        try:
            ast.parse(result)
        except SyntaxError:
            lines = result.split('\n')
            filtered = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    filtered.append(line)
                    continue
                try:
                    ast.parse(stripped)
                    filtered.append(line)
                except SyntaxError:
                    logger.debug(f'[LMP "{self._name}"] filtered non-code line: {stripped[:80]}')
            new_result = '\n'.join(filtered).strip()
            n_removed = len(lines) - len(filtered)
            if n_removed > 0:
                logger.warning(
                    f'[LMP "{self._name}"] filtered {n_removed} '
                    f'natural-language line(s) from LLM output'
                )
            # Use filtered result even if empty (model produced no valid code)
            result = new_result
            if not result:
                logger.warning(f'[LMP "{self._name}"] no valid Python code after filtering')

        # Fix indentation leaking from CoT reasoning (e.g. Qwen-Thinking)
        if result:
            dedented = textwrap.dedent(result).strip()
            if dedented != result.strip():
                logger.debug(f'[LMP "{self._name}"] dedented code output')
                result = dedented

        # Fix broken indentation: lines indented without a preceding control statement
        if result:
            try:
                ast.parse(result)
            except SyntaxError:
                lines = result.split('\n')
                fixed = []
                for i, line in enumerate(lines):
                    if not line.strip():
                        fixed.append(line)
                        continue
                    indent = len(line) - len(line.lstrip())
                    prev_indent = len(fixed[-1]) - len(fixed[-1].lstrip()) if fixed and fixed[-1].strip() else 0
                    # Line jumps in indent without a control statement on previous line
                    if indent > prev_indent and fixed:
                        prev_stripped = fixed[-1].rstrip()
                        if prev_stripped and not prev_stripped.endswith(':'):
                            # Dedent this line to match previous level
                            fixed.append(' ' * prev_indent + line.lstrip())
                            continue
                    fixed.append(line)
                new_result = '\n'.join(fixed)
                try:
                    ast.parse(new_result)
                    logger.warning(f'[LMP "{self._name}"] fixed broken indentation')
                    result = new_result
                except SyntaxError:
                    pass  # keep original if fix didn't help

        return result

    @staticmethod
    def _encode_image(img_array):
        """Encode numpy RGB array to base64 JPEG string."""
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _cached_api_call(self, **kwargs):
        user1 = kwargs.pop('prompt')
        # GLM / gpt-oss / DeepSeek models repeat context (objects list, # Query:)
        # triggering stop tokens prematurely. Remove all stop tokens; rely on max_tokens.
        model_name = kwargs.get('model', '')
        if 'glm' in model_name.lower() or 'gpt-oss' in model_name or 'gpt_oss' in model_name or 'deepseek' in model_name.lower():
            kwargs['stop'] = []
        new_query = '# Query:' + user1.split('# Query:')[-1]
        # Preserve '# Query:' in few-shot examples (only remove the last one which is the actual query)
        parts = user1.split('# Query:')
        user1 = '# Query:'.join(parts[:-1]).strip()
        # Model-specific prompt additions from config
        model_name = kwargs.get('model', '')
        extra_instruction = self._cfg.get('prompt_extra', '')
        if not extra_instruction:
            # Check model-specific overrides
            model_prompts = self._cfg.get('model_prompts', {})
            for pattern, instruction in model_prompts.items():
                if pattern.lower() in model_name.lower():
                    extra_instruction = ' ' + instruction
                    break
        else:
            extra_instruction = ' ' + extra_instruction
        # Reasoning models (Qwen3 with thinking, etc.) tend to over-engineer code with
        # undefined runtime flags. Add an explicit no-conditional instruction for those.
        thinking_active = ('qwen3' in model_name.lower() and 'instruct-2507' not in model_name.lower()
                           and os.environ.get('QWEN3_ENABLE_THINKING', '0') == '1')
        literal_only_clause = (
            " CRITICAL: emit only literal Python values (numbers, strings) inside function calls; "
            "do NOT use conditional expressions referencing undefined runtime flags like "
            "`near_human`, `near_cat`, or `is_X`. The provided helper functions accept fixed numeric values, "
            "not boolean-conditioned expressions."
        ) if thinking_active else ""
        user1 = f"I would like you to help me write Python code to control a mobile robot navigating safely in a kitchen environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay).{literal_only_clause}{extra_instruction} I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
        assistant1 = f'Got it. I will complete what you give me next.'
        user2 = new_query
        # handle given context (this was written originally for completion endpoint)
        if user1.split('\n')[-4].startswith('objects = ['):
            obj_context = user1.split('\n')[-4]
            # remove obj_context from user1
            user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
            # add obj_context to user2
            user2 = obj_context.strip() + '\n' + user2
        # Use 'developer' role for GPT-oss harmony channel compatibility;
        # standard models treat 'developer' same as 'system'
        from utils.utils import load_prompt
        sys_content = load_prompt('robocasa_navigation_system/default_system_prompt.txt').strip()
        system_prompt_extra = self._cfg.get('system_prompt_extra', '')
        if system_prompt_extra:
            sys_content += '\n\n' + system_prompt_extra
        model_name = kwargs.get('model', '')
        if 'gpt-oss' in model_name or 'gpt_oss' in model_name:
            sys_role = 'developer'
            sys_content += " ONLY use functions shown in the examples. Do NOT invent new function signatures or keyword arguments. Call execute_navigation() EXACTLY as shown in examples — each map argument must be a SINGLE function call, never a list. Combine multiple avoidance constraints into ONE get_avoidance_map() call with a single descriptive string."
        else:
            sys_role = 'system'
        # Cosmos-Reason2 model card requires explicit <think>/<answer> output format
        # for the reasoning-traced answer path. Without it, reasoning is partially gated.
        if 'cosmos' in model_name.lower():
            sys_content += (
                "\n\nAnswer the question in the following format: "
                "<think>\nyour reasoning\n</think>\n\n"
                "<answer>\nyour answer\n</answer>"
            )
        # Build the last user message: multimodal if model is VLM and images are available
        attach_images = self._images and is_vlm(model_name)
        if attach_images:
            # Add VLM image-usage instruction to the context prompt
            vlm_instruction = (
                "\n\nCamera images of the current kitchen scene are attached below. "
                "Use them to understand the spatial layout, object locations, and potential hazards. "
                "Output ONLY Python code — do NOT describe the images."
            )
            user1 += vlm_instruction
            # Llama-3.2-Vision requires images BEFORE text; other VLMs accept text-first
            images_first = 'llama-3.2' in model_name.lower() and 'vision' in model_name.lower()
            cam_labels = self._image_labels or [f"camera_{i}" for i in range(len(self._images))]
            # Llama-3.2-Vision (mllama) only supports 1 image per request — filter to topview only
            mllama_one_image = 'llama-3.2' in model_name.lower() and 'vision' in model_name.lower()
            images_to_use = list(self._images)
            labels_to_use = list(cam_labels)
            if mllama_one_image and len(images_to_use) > 1:
                # prefer topview if present, otherwise first image
                tv_idx = next((i for i, l in enumerate(labels_to_use) if 'topview' in (l or '').lower()), 0)
                images_to_use = [images_to_use[tv_idx]]
                labels_to_use = [labels_to_use[tv_idx]]
            image_blocks = []
            for img, label in zip(images_to_use, labels_to_use):
                b64 = self._encode_image(img)
                image_blocks.append({"type": "text", "text": f"[{label}]"})
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            if images_first:
                user2_content = image_blocks + [{"type": "text", "text": user2}]
            else:
                user2_content = [{"type": "text", "text": user2}] + image_blocks
            logger.debug(f'[LMP "{self._name}"] attaching {len(self._images)} image(s) to request: {cam_labels} (images_first={images_first})')
        else:
            user2_content = user2
        messages=[
            {"role": sys_role, "content": sys_content},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2_content},
        ]
        kwargs['messages'] = messages
        # Thinking mode for Qwen3 hybrid models (not Instruct-2507 which has no thinking)
        # Default: disabled. Override via QWEN3_ENABLE_THINKING=1 env var.
        extra_body = None
        if 'qwen3' in model_name.lower() and 'instruct-2507' not in model_name.lower():
            thinking_enabled = os.environ.get('QWEN3_ENABLE_THINKING', '0') == '1'
            extra_body = {"chat_template_kwargs": {"enable_thinking": thinking_enabled}}
            logger.debug(f'[LMP "{self._name}"] Qwen3 enable_thinking={thinking_enabled}')
        elif 'glm' in model_name.lower() and '4.6v' in model_name.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            logger.debug(f'[LMP "{self._name}"] disabling GLM-4.6V thinking mode')
        elif 'gpt-oss' in model_name or 'gpt_oss' in model_name:
            extra_body = {"reasoning_effort": "low"}
            logger.debug(f'[LMP "{self._name}"] setting GPT-oss reasoning_effort=low')
        create_kwargs = dict(kwargs)
        if extra_body:
            create_kwargs['extra_body'] = extra_body

        # Disk cache (text-only models only — VLM image bytes vary each call
        # and JSON-serialising them is wasteful). Key derived from full
        # messages + sampling params + extra_body so different prompts /
        # temperatures / thinking modes are stored separately.
        cache = _get_cache() if not attach_images else None
        cache_key = None
        if cache is not None:
            try:
                cache_key = {
                    'model': model_name,
                    'messages': messages,
                    'temperature': kwargs.get('temperature'),
                    'max_tokens': kwargs.get('max_tokens'),
                    'stop': kwargs.get('stop'),
                    'extra_body': extra_body,
                }
                # JSON-serializable check (skips weird types)
                import json as _json
                _json.dumps(cache_key)
            except Exception:
                cache_key = None
            if cache_key is not None and cache_key in cache:
                logger.debug(f'[LMP "{self._name}"] cache HIT')
                cached = cache[cache_key]
                return self._extract_code(_CachedMsg(
                    content=cached.get('content', ''),
                    reasoning_content=cached.get('reasoning_content'),
                ))

        ret = self._client.chat.completions.create(**create_kwargs)
        msg = ret.choices[0].message
        logger.debug(f'[LMP "{self._name}"] raw response ({len(msg.content)} chars): {msg.content[:500]}')

        # Store in cache (after successful API call)
        if cache is not None and cache_key is not None:
            try:
                cache[cache_key] = {
                    'content': msg.content or '',
                    'reasoning_content': getattr(msg, 'reasoning_content', None),
                }
                logger.debug(f'[LMP "{self._name}"] cache STORE')
            except Exception as _ce:
                logger.warning(f'[LMP "{self._name}"] cache store failed: {_ce}')

        return self._extract_code(msg)

    def __call__(self, *queries, **kwargs):
        # Accept multiple positional args (e.g. LLM calls lmp('q1', 'q2', 'q3')) and join them
        query = ' '.join(str(q) for q in queries)
        prompt, user_query = self.build_prompt(query)

        # Model-specific max_tokens override (some models have small context, need shorter output budget)
        model_name_lc = (self._cfg.get('model') or '').lower()
        max_tokens = self._cfg['max_tokens']
        if 'deepseek-vl2' in model_name_lc:
            # DeepSeek-VL2 max_pos=4096; with prompt ~750 tokens, leave room → cap output at 1024
            max_tokens = min(max_tokens, 1024)
        if 'llama-3.2' in model_name_lc and 'vision' in model_name_lc:
            # Llama-3.2-Vision max_pos=4096 (eval config); with prompt ~700 tokens, leave room → cap at 1024
            max_tokens = min(max_tokens, 1024)
        start_time = time.time()
        last_err = None
        for api_attempt in range(API_MAX_RETRIES):
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=max_tokens
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(f'API error (attempt {api_attempt+1}/{API_MAX_RETRIES}): {e} — retrying in 3s')
                sleep(3)
        else:
            raise LMPApiUnreachable(
                f'[LMP "{self._name}"] API unreachable after {API_MAX_RETRIES} attempts: {last_err}'
            )
        logger.info(f'[LMP "{self._name}"] API call {time.time() - start_time:.2f}s')

        if not code_str.strip():
            raise LMPEmptyOutput(
                f'[LMP "{self._name}"] empty LLM output for query: {query[:100]}'
            )

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        # Persist generated code at INFO level so it lands in the per-task
        # run.log — useful for analysing which affordance/avoidance the LMP
        # actually emitted. Plain (un-highlighted) so terminal escape codes
        # don't clutter the file.
        _hdr = (f'## "{self._name}" generated code'
                + (f'  (context: "{self._context}")'
                   if self._cfg['include_context'] else ''))
        logger.info('#'*40 + f'\n{_hdr}\n' + '#'*40 + f'\n{to_log}\n' + '#'*40)
        logger.debug(to_log_pretty)  # keep pretty highlight for -v console

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            if not to_exec.strip():
                return None
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars, lmp_name=self._name)
            except Exception as e:
                logger.error(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars, lmp_name=self._name)

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
            result = lvars[self._cfg['return_val_name']]
            # Wrap callable map returns so LLM can combine them with numpy ops (e.g. map_a + map_b)
            if callable(result) and self._name not in ['parse_query_obj']:
                return DeferredMap(result)
            return result


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None, lmp_name=None):
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
    except bdb.BdbQuit:
        raise
    except Exception as e:
        import traceback as _tb
        logger.error(f'Error executing code:\n{code_str}')
        logger.error(f'Error message:\n{e}')
        logger.error(f'Traceback:\n{_tb.format_exc()}')
        # Attach a per-LMP frame so outer handlers can persist the
        # generated code and originating LMP into failure_message.
        # Innermost frame appended first; outer LMPs append as it bubbles up.
        if not hasattr(e, '_lmp_code_chain'):
            e._lmp_code_chain = []
        # Truncate per-frame to keep results.json bounded.
        e._lmp_code_chain.append({
            'lmp': lmp_name or '?',
            'code': code_str if len(code_str) < 2000 else code_str[:2000] + '...[truncated]',
        })
        raise