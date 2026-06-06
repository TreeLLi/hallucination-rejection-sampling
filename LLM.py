import torch as tc
from transformers import AutoTokenizer, AutoModelForCausalLM

from addict import Dict
from enum import Enum
import copy

from utils import log_w_indent, split_sentences

from typing import Optional, Tuple, List
from transformers.generation.logits_process import (
    SuppressTokensLogitsProcessor,
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
)

MODELS = {
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3-32b': 'Qwen/Qwen3-32B',
    'qwen3-4b-instruct': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-30b-instruct': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    'qwen3-235b': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'llama3-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'llama4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'hunyuan-a13b': 'tencent/Hunyuan-A13B-Instruct',
    'gemma3-27b': 'google/gemma-3-27b-it',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
    'glm4.5-air': 'zai-org/GLM-4.5-Air',
    'glm4-32b': 'zai-org/GLM-4-32B-0414',
    'glm4-9b': 'zai-org/GLM-4-9B-0414',
    'mistral3-24b': 'mistralai/Mistral-Small-24B-Instruct-2501'
}

MULTIMODAL = [
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'google/gemma-3-27b-it',
    'mistralai/Mistral-Small-3.2-24B-Instruct-2506'
]

MODEL_NAMES = sorted(list(MODELS.keys()))

CACHED_MODELS = {}

def get_model(model_name, device='cuda'):
    model_name = model_name.lower()
    model_name = MODELS[model_name]
    
    if model_name in CACHED_MODELS:
        model = CACHED_MODELS[model_name]
    else:
        # init model
        model_class = Qwen3 if 'Qwen3' in model_name else HFTransformer
        model = model_class(model_name, device)
        CACHED_MODELS[model_name] = model

    return model

class LLM:

    def __init__(self, model_name):
        self.model_name = model_name

    def format_message(self, role, content):
        if self.model_name in MULTIMODAL:
            return {
                'role': role,
                'content': [{
                    'type': 'text',
                    'text': content
                },]
            }
        else:
            return {
                'role': role,
                'content': content
            }
        
        
    def format_messages(self, user_prompt=None, syst_prompt=None, msg_history=None):
        if syst_prompt and msg_history:
            for msg in msg_history:
                if msg['role'] != 'system': continue
                raise Exception("Duplicate system prompt!")
            
        messages = []
        if syst_prompt:
            message = self.format_message('system', syst_prompt)
            messages.append(message)
            
        if msg_history: messages += msg_history

        if user_prompt:
            message = self.format_message('user', user_prompt)
            messages.append(message)

        return messages


class Buffer:

    def __init__(self, device):
        self.device = device
        self.reset()
        
    def reset(self):
        self.tokens = tc.empty(0, dtype=tc.long).to(self.device)
        self.text = ''
        
    def add(self, token, text):
        self.tokens = tc.cat([self.tokens, token.unsqueeze(-1)], dim=-1)
        self.text += text

class SuppressEOS(Enum):
    BEGIN = 1
    ALL = 2
    NOT = 0

import re

def is_sentence_valid(text: str) -> bool:
    """
    Check if the given string is a valid sentence.
    
    A valid sentence must:
      - Start with a capital letter (A-Z).
      - End with '.', '?', or '!'.
      - Contain at least one word character in between.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    
    # Regex: start with capital, something in between, ends with punctuation
    pattern = r'^[A-Z][^!?]*[.?!]$'
    
    if re.match(pattern, text.strip()):
        return True
    return False

    
class HFTransformer(LLM):

    def __init__(self, model_name, device='cuda'):
        super().__init__(model_name=model_name)
        self.device = device        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.eos_token = tc.tensor([self.tokenizer.eos_token_id]).to(device)
        
    def format_messages(self,
                        user_prompt=None,
                        syst_prompt=None,
                        msg_history=None,
                        add_generation_prompt=True,
                        return_str=True,
                        thinking=False,
                        **args):
        # by default, thinking is ignored
        messages = super().format_messages(user_prompt=user_prompt,
                                           syst_prompt=syst_prompt,
                                           msg_history=msg_history)
        if return_str:
            messages = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=False,
                                                          add_generation_prompt=add_generation_prompt,
                                                          **args)
        return messages
        
    def tokenize(self, text):
        tokenized = self.tokenizer([text], return_tensors="pt").to(self.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def decode(self, tokens, skip_special_tokens=True):
        tokens = tokens.tolist() if isinstance(tokens, tc.Tensor) else tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def cache_prompt(self, user_prompt=None, syst_prompt=None, add_generation_prompt=False, return_tokens=False):
        messages = self.format_messages(
            user_prompt=user_prompt,
            syst_prompt=syst_prompt,
            add_generation_prompt=add_generation_prompt
        )
        tokens, attention_mask = self.tokenize(messages)
        output = self.model(input_ids=tokens, attention_mask=attention_mask, use_cache=True)

        if return_tokens:
            return output.past_key_values, tokens, attention_mask
        else:
            return output.past_key_values
    
    def generation_configs(self, **args):
        args = Dict(args)

        args.temperature = float(args.get("temperature") or 0.0)
        args.top_k = int(args.get("top_k") or 0)
        args.top_p = float(args.get("top_p") or 1.0)
        args.num_beams = int(args.get("num_beams") or 1)
        args.num_beam_groups = int(args.get("num_beam_groups") or 1)
        args.diversity_penalty = float(args.get("diversity_penalty") or 0.0)
        args.min_p = float(args.get("min_p") or 0.0)

        args.do_sample = args.temperature != 0.0

        if args.get('do_sample') is True:
            # sampling path must NOT use group beams/diversity
            args.num_beam_groups = 1
            args.diversity_penalty = 0.0
            args.num_beams = 1 if args.get('num_beams') in (None, 0) else args.num_beams
        else:
            # deterministic beams: allow group beams/diversity
            if args.get('num_beam_groups', 1) > 1 or (args.get('diversity_penalty') or 0) > 0:
                args.do_sample = False 
                args.temperature = None

        if not getattr(self.model.generation_config, 'pad_token_id', None):
            args.pad_token_id = self.tokenizer.eos_token_id

        if args.past_key_values != {} and not args.past_key_values:
            args.pop('past_key_values', None)
            
        return args

    def next_token(self,
                   tokens,
                   attention_mask=None,
                   past_key_values=None,
                   do_sample=True,
                   temperature=0.6,
                   top_k=50,
                   top_p=1,
                   diversity_penalty=None,
                   num_beams=None,
                   num_beam_groups=None,
                   min_p=None,
                   suppress_tokens=None):

        args = self.generation_configs(do_sample=do_sample,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p,
                                       diversity_penalty=diversity_penalty,
                                       num_beams=num_beams,
                                       num_beam_groups=num_beam_groups,
                                       min_p=min_p,
                                       past_key_values=past_key_values)
        
        output = self.model.generate(input_ids=tokens,
                                     attention_mask=attention_mask,
                                     max_new_tokens=1,
                                     output_logits=True,
                                     return_dict_in_generate=True,
                                     suppress_tokens=suppress_tokens,
                                     **args)
        
        next_token = output.sequences[0][-1:]

        return next_token, output.past_key_values, output.logits[0]


    def sample(self,
               input_tokens: tc.LongTensor,
               logits: tc.LongTensor,
               temperature: float = 1.0,
               top_k: int = 50,
               top_p: float = 1.0,
               diversity_penalty=None,
               num_beams=None,
               num_beam_groups=None,
               min_p=None,
               suppress_tokens=None,
    ):
        """
        Sample the next token using Hugging Face LogitsWarper classes, with KV cache support.

        Args:
            model: A Hugging Face AutoModelForCausalLM (already on device). Should be decoder-only.
            input_tokens (LongTensor): [batch, seq_len] current input IDs.
            temperature (float): temperature (>0). Use 1.0 to skip.
            top_k (int): keep top_k tokens (0 to disable).
            top_p (float): nucleus probability threshold in (0,1], 1.0 to disable.

        Returns:
            next_token_id (LongTensor): [batch, 1] sampled token id.
        """

        # Build logits warpers
        warpers = LogitsProcessorList()
        if suppress_tokens is not None:
            warpers.append(SuppressTokensLogitsProcessor(suppress_tokens, device=input_tokens.device))
        if temperature and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature=temperature))
        if top_k and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
            
        # Apply warpers (temperature, top-k, top-p)
        warped_logits = warpers(input_tokens, logits) if len(warpers) > 0 else logits

        # Softmax -> sample
        probs = tc.softmax(warped_logits, dim=-1)
        next_token_id = tc.multinomial(probs, num_samples=1)

        return next_token_id[0]            
            
    def next_sentence(self, 
                      input_tokens=None,
                      attention_mask=None,
                      pre_text='',
                      past_key_values=None,
                      logits=None,
                      max_new_tokens=100,
                      do_sample=True,
                      temperature=0.6,
                      suppress_EOS=SuppressEOS.NOT,
                      sent_space_required=False,
                      abbrevs=None,
                      top_k=50,
                      top_p=1,
                      diversity_penalty=None,
                      num_beams=None,
                      num_beam_groups=None,
                      min_p=None,
                      log=False,
                      log_indent=2,
                      bad_words_ids=None):
        tokens_sofar = input_tokens
        out_tokens, out_text, out_logits = [], [], []
        check_sent = False
        post_text = ''

        suppress_tokens = []
        EOS_token = self.eos_token.tolist()[0]
        if suppress_EOS != SuppressEOS.NOT: suppress_tokens.append(EOS_token)

        sent_space_tokens = []
        
        for i in range(max_new_tokens):
            # only suppress EOS at the first token
            if i != 0 and suppress_EOS != SuppressEOS.ALL and EOS_token in suppress_tokens:
                suppress_tokens.remove(EOS_token)
            
            if i == 0 and logits is not None:
                # reuse the logits to sample when rejected and resampled.
                # TODO: right now this does not include diversity penality or beam search
                next_token = self.sample(tokens_sofar,
                                         logits,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         suppress_tokens=suppress_tokens)
            else:
                attention_mask = tc.ones_like(tokens_sofar, dtype=attention_mask.dtype)
                next_token, past_key_values, logits = self.next_token(
                    tokens_sofar,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    diversity_penalty=diversity_penalty,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_p=min_p,
                    suppress_tokens=suppress_tokens,
                )

            next_text = self.decode(next_token)
            
            tokens_sofar = tc.cat([tokens_sofar, next_token.unsqueeze(-1)], dim=-1)
            out_tokens.append(next_token.unsqueeze(-1))
            out_logits.append(logits)
            
            if next_token == self.eos_token: break
            
            out_text.append(next_text)
            
            has_delimiter = any([c in next_text for c in ('.', '!', '?')])
            # check the token with delimiter and its immediate next token
            if not (has_delimiter or check_sent): continue
            
            check_sent = has_delimiter
            text_sofar = pre_text + ''.join(out_text)

            sents = split_sentences(text_sofar, extra_abbrevs=abbrevs)
            
            # a complete sentence is identified if the text is split into more than 1 part
            if len(sents) <= 1: continue                
                
            if len(sents) > 2:
                print(f"The new token results in more than 1 complete sentence: {sents}.")
                print("Keep the first sentence and discard the rest.")

            sent = sents[0].strip()
            text = pre_text
            # find the idx where sentence is split
            for j, _text in enumerate(out_text, start=1):
                text += _text
                if sent in text: break
                
            is_last_token_split = j == len(out_text)
            
            discarded_text = ''.join(out_text[j:])
            print(f"Discarded text: {discarded_text}")
            
            out_text = out_text[:j]
            out_tokens = out_tokens[:j]
            out_logits = out_logits[:j+1]
            # get the post sentence text in the last token
            end_idx = text.find(sent) + len(sent)
            if end_idx < len(text) and text[end_idx] in ('"', "'", '*'):
                end_idx += 1
            while end_idx < len(text):
                if text[end_idx] in ('\n', ' '):
                    end_idx += 1
                else:
                    post_text = text[end_idx:]
                    print(f"tail: `{post_text}`")
                    break
                    
            cache_length = input_tokens.size(1) + len(out_tokens)
            if is_last_token_split:
                cache_length -= 1
                out_logits.append(None)
                
            past_key_values.crop(cache_length)
            
            break

        sent = pre_text + ''.join(out_text)
        if len(post_text) > 0: sent = sent[:-len(post_text)]
        out_tokens = tc.cat(out_tokens, dim=-1)
        logits = Dict({
            'pre': out_logits[0],
            'post': out_logits[-1]
        })
        return sent, post_text, out_tokens, past_key_values, logits
    
        
    def complete(self,
                 user_prompt=None,
                 syst_prompt=None,
                 msg_history=None,
                 input_tokens=None,
                 attention_mask=None,
                 thinking=False,
                 past_key_values=None,
                 max_new_tokens=10000,
                 return_cache=False,
                 return_tokens=False,
                 log=False,
                 log_indent=2,
                 **args):

        assert user_prompt is not None or input_tokens is not None
        if user_prompt:
            messages = self.format_messages(user_prompt=user_prompt,
                                            syst_prompt=syst_prompt,
                                            msg_history=msg_history,
                                            thinking=thinking)
            
            input_tokens, attention_mask = self.tokenize(messages)
            
        args = self.generation_configs(past_key_values=past_key_values, **args)
        
        if log: log_w_indent(f'Input: {user_prompt}', log_indent)
        
        output = self.model.generate(input_ids=input_tokens,
                                     attention_mask=attention_mask,
                                     max_new_tokens=max_new_tokens,
                                     return_dict_in_generate=return_cache,
                                     **args)
        all_tokens = output.sequences if return_cache else output
        out_tokens = all_tokens[0][input_tokens.size(1):]
        out_str = self.decode(out_tokens)

        if log: log_w_indent(f'Output: {out_str}', log_indent, symbol='xx')

        gen = (out_str, out_tokens) if return_tokens else out_str
        
        if return_cache: return gen, output.past_key_values
        else: return gen


class Qwen3(HFTransformer):

    def __init__(self, model_name, device='cuda'):
        super().__init__(model_name=model_name, device=device)

        self.thinking_tokens = (151667, 151668)

    def format_messages(self,
                        user_prompt,
                        syst_prompt=None,
                        msg_history=None,
                        add_generation_prompt=True,
                        return_str=True,
                        thinking=False):
        messages = super().format_messages(
            user_prompt=user_prompt,
            syst_prompt=syst_prompt,
            msg_history=msg_history,
            add_generation_prompt=add_generation_prompt,
            return_str=return_str,
            enable_thinking=thinking
        )
        return messages
    
    def complete(self,
                 thinking=False,
                 return_cache=False,
                 return_tokens=False,
                 return_think=False,
                 **args):
        output = super().complete(
            thinking=thinking,
            return_tokens=return_tokens,
            return_cache=return_cache,
            **args
        )
        if not thinking: return output
        
        if return_cache:
            output, cache = output
            
        out_str = output[0] if isinstance(output, tuple) else output
        out_str = out_str.split('</think>')
        think_str = out_str[0].replace('<think>', '')
        final_str = out_str[1] if len(out_str) > 1 else ''

        if return_tokens:
            think_start, think_end = self.thinking_tokens
            out_tokens = output[1]
            think_end_idx = out_tokens[0].tolist().index(think_end)
            think_tokens = out_tokens[:, 1: think_end_idx]
            final_tokens = out_tokens[:, think_end_idx+1:] if len(out_str) > 1 else []
            
            think = (think_str, think_tokens)
            final = (final_str, final_tokens)

            assert think_str == self.model.decode(think_tokens)
            assert final_str == self.model.decode(final_tokens)
        else:
            think = think_str
            final = final_str

        rtn = [final, think] if return_think else [final]
        if return_cache: rtn.append(cache)

        return rtn[0] if len(rtn) == 1 else rtn
