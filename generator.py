import re
import torch as tc
from enum import Enum
from warnings import warn
import copy

from uncertainty import ESTIMATORS

from utils import split_sentences, is_non_answer
from LLM import SuppressEOS


rewrite_syst_prompt = f'''You are given a query and a list of fact claims. Each fact claim describes one fact. Your task is to generate a new single sentence that incorporates the facts described in the fact claims.

### Instructions

1. Identify the fact in each fact claim.
2. Remove any facts that are irrelevant to the query or to the other facts.
3. If relevant facts remain, combine them into a single new sentence.
4. The subject of the output sentence must be the same as the subject in the fact claims (e.g., the specific name or pronouns like He/She/It/This/That).
4. If no relevant facts remain, return an empty string.

### Output Requirements

* Output only the generated sentence, and nothing else.
* The output must be exactly one complete, grammatically correct sentence.
* Do not invent new facts or add extra information beyond the fact claims.
* If no facts remain, output an empty string.

---

### Example 1

#### Input:
Query: Tell me about Ling Li.

Fact claims:
- Ling is an AI researcher.
- He is known for works in AI safety.

#### Output:
Lin is an AI researcher, known for works in AI safety.

---

### Example 2

#### Input:
Query: Tell me about Lin Li.

Fact claims:
- He is a Chinese.
- He is an AI researcher.

#### Output:
He is a Chinese AI researcher.

---

### Example 3

#### Input:
Query: Tell me about Li Lin.

Fact claims:
- She is known for education.
- Oxford is located in UK.

#### Output:
She is known for education.

'''


class BaseGenerator:

    def __init__(self, model, device):
        self.device = device
        self.model = model
        
class NaiveGenerator(BaseGenerator):
    def generate(
            self,
            user_prompt,
            syst_prompt,
            max_new_tokens,
            temperatures,
            **args
    ):
        del args['entity']
        temperature = temperatures[0]
        do_sample = temperature != 0.0
        return self.model.complete(user_prompt=user_prompt,
                                   syst_prompt=syst_prompt,
                                   max_new_tokens=max_new_tokens,
                                   do_sample=do_sample,
                                   temperature=temperature,
                                   **args)



def is_prop_hallued(prop, uncertainty_threshold, estimator, entail_check=True):
    hallu = prop.uncertainty > uncertainty_threshold
    if hallu or not entail_check: return hallu
    
    # use the cluster with minimum uncertainty
    min_q_uct, answers, clusters, expected_answer = 1e6, None, None, None
    for question, data in prop.qas.items():
        q_uct = data.uncertainty
        if q_uct > min_q_uct: continue
        
        min_q_uct = q_uct
        answers = data.answers
        clusters = data.clusters
        expected_answer = data.expected_answer

    entailed, cluster_answer = estimator.is_entailed_largest_cluster(expected_answer,
                                                                     answers,
                                                                     clusters)
    return not entailed
    
class Hallu(Enum):
    ALL = 1
    NONE = 0
    PART = 2

    
def is_sentence_hallu(props):
    n_props = len(props)
    n_hallu = len([prop for prop in props if prop.hallued])

    if n_hallu == 0: return Hallu.NONE
    elif n_hallu == n_props: return Hallu.ALL
    else: return Hallu.PART
    
def reset_cache(logits, past_key_values, tokens_sofar):
    # resample the first token with the existing logits
    logits = logits if logits is None else logits.pre
    # keep the cache up to the pre sentence token.
    if past_key_values is not None:
        seq_len = tokens_sofar.size(1)
        past_key_values = past_key_values.crop(seq_len)

    return logits, past_key_values


class UncertaintyGenerator(BaseGenerator):
    
    def __init__(self,
                 model,
                 uncertainty_estimator,
                 uncertainty_base_model,
                 uncertainty_model_thinking,
                 uncertainty_threshold,
                 max_resample_attempts,
                 max_repetition_attempts,
                 rewrite,
                 resample_policy,
                 entail_check=False,
                 device='cuda',
                 temperatures=[],
                 top_p=None,
                 top_k=None,
                 **kwargs):
        super().__init__(model, device)
        
        # init uncertainty estimator
        self.uct_estimator = ESTIMATORS[uncertainty_estimator](
            model=uncertainty_base_model,
            thinking=uncertainty_model_thinking,
            temperature=temperatures[0],
            top_p=top_p,
            top_k=top_k,
            device=device,
            **kwargs
        )
        
        # init corrector
        self.rewrite = rewrite
        self.writer = model
        self.rewrite_cache = None
        
        self.resample_policy = resample_policy
        
        self.uncertainty_threshold = uncertainty_threshold
        self.max_resample_attempts = max_resample_attempts
        self.max_repetition_attempts = max_repetition_attempts
        self.entail_check = entail_check
        
    def add_hallu_aspects_to_syst_prompt(self, syst_prompt, aspects):
        if len(aspects) == 0: return syst_prompt
        
        aspects = [f'- {aspect}' for aspect in aspects]

        syst_prompt += '\n\n' + '\n'.join(
            [
                "The output must not include the answer to the following questions:",
                *aspects,
            ]
        )
        return syst_prompt
    
    def generate(self,
                 user_prompt,
                 syst_prompt,
                 entity,
                 thinking=False,
                 max_new_tokens=500,
                 temperatures=[1],
                 **args):
        messages = self.model.format_messages(user_prompt, syst_prompt)
        
        input_tokens, attention_mask = self.model.tokenize(messages)
        past_key_values, logits = None, None
        
        tokens_sofar = input_tokens
        text_sofar, pre_text = '', ''
        rest_new_tokens = max_new_tokens
        
        fact_sents_tokens = []
        fact_sents_text = []
        
        hallued_sents_text = []
        hallued_props = []

        all_sents = []
        
        nattempt = 1
        
        suppress_EOS = False
        last_fact_sent_suppress_EOS = False
        end_gen = False
        
        # TODO: words w periods should not be split.
        abbrevs = entity if '.' in entity else None
        
        n_repetition = 0

        try:
            while rest_new_tokens > 0:

                if self.resample_policy == 'decode':
                    t_idx = max(n_repetition, nattempt-1)
                else:
                    t_idx = n_repetition
                temperature = temperatures[t_idx] if t_idx < len(temperatures) else temperatures[-1]

                if text_sofar != '':
                    decoded_text = self.model.decode(tokens_sofar[0][input_tokens.size(1):])

                    if text_sofar+pre_text != decoded_text:
                        raise Exception(f"Decoded text mismatched the test sofar:\nDecoded: {decoded_text}\nText: {text_sofar}")

                sent_space_required = text_sofar != '' and text_sofar[-1] not in (' ', '\n')
                sent_text, post_text, sent_tokens, past_key_values, logits = self.model.next_sentence(
                    tokens_sofar,
                    attention_mask,
                    pre_text=pre_text,
                    past_key_values=past_key_values,
                    logits=logits,
                    max_new_tokens=rest_new_tokens,
                    sent_space_required=sent_space_required,
                    temperature=temperature,
                    suppress_EOS=suppress_EOS,
                    **args
                )

                if is_non_answer(sent_text):
                    print(f"[END] Generation abort. The model has no knowledge about {entity}.")
                    break

                normalized_sent_text = sent_text.strip().lower()
                if normalized_sent_text in all_sents:
                    print(f"Resample as repeated sentence: {sent_text}")
                    logits, past_key_values = reset_cache(logits, past_key_values, tokens_sofar)
                    n_repetition += 1
                    if n_repetition == self.max_repetition_attempts:
                        print("[END] Generation abort due to reaching the max number of repetition attempts.")
                        break
                    continue
                else:
                    all_sents.append(normalized_sent_text)
                    n_repetition = 0

                print(f"Verified text:\t{text_sofar}")
                print(f"Attempting {nattempt}/{self.max_resample_attempts} with T={temperature:1.2f}: \t{sent_text}")

                fact_sofar = ''.join(fact_sents_text) if fact_sents_text != [] else ''
                props = self.uct_estimator.get_sent_uncertainty(
                    user_query=user_prompt,
                    text_sofar=fact_sofar,
                    sentence=sent_text,
                    query_entity=entity,
                )

                if not props:
                    print("No factual propositions are found in the sentence.")
                    print("Discard the current sentence and resample.")
                    logits, past_key_values = reset_cache(logits, past_key_values, tokens_sofar)
                    continue

                if nattempt == 1:
                    # the begin of the next sentence is already sampled by token or post-sent text.
                    suppress_EOS = logits.post is not None or post_text != ''
                    # suppress EOS for next sentence by BEGIN, but allow EOS for next next sentence
                    suppress_EOS = SuppressEOS.BEGIN if suppress_EOS else Suppress.NOT
                    last_fact_sent_suppress_EOS = suppress_EOS

                # end generation if EOS ever detected
                end_gen = end_gen or (sent_tokens[0][-1] == self.model.eos_token)

                for i, prop in enumerate(props):
                    prop.hallued = is_prop_hallued(prop,
                                                   self.uncertainty_threshold,
                                                   self.uct_estimator,
                                                   self.entail_check)

                    if prop.hallued:
                        hallued = 'Hallued'
                        hallued_props.append(prop)
                    else:
                        hallued = 'Factual'

                    print(f"Prop-{i} (Uncertainty: {prop.uncertainty: .3f}, {hallued}) :\t{prop.fact_claim}\t({prop.entity})")

                hallu_state = is_sentence_hallu(props)
                if hallu_state == Hallu.NONE:
                    print("All propositions are factual, keeping the sentence.")
                elif hallu_state == Hallu.ALL:
                    print("All propositions are hallucinated.")
                else:
                    print("Factual propositions exist, but not all.")

                keep_sentence = hallu_state == Hallu.NONE

                if hallu_state == Hallu.PART and self.rewrite:
                    stripped_sent_text = sent_text.strip()
                    print("Rewrite the original sentence:")
                    print(f"    {stripped_sent_text}")

                    rewritten_sent_text = self.rewrite_sentence(
                        query=user_prompt,
                        sent=stripped_sent_text,
                        props=props,
                        temperature=temperatures[0], # TODO: main args
                        **args
                    )

                    if rewritten_sent_text == "":
                        print("Failed to rewrite into a valid sentence")
                        keep_sentence = False
                    else:
                        print(f"to the new sentence:")
                        print(f"    {rewritten_sent_text}")

                        # restore the original leading and tail spacing
                        rewritten_sent_text = sent_text.replace(stripped_sent_text, rewritten_sent_text)

                        normalized_sent_text = rewritten_sent_text.strip().lower()
                        if normalized_sent_text in all_sents:
                            print("Discard the rewritten sentence as it already sampled.")
                            keep_sentence = False
                        else:
                            all_sents.append(normalized_sent_text)
                            sent_text = rewritten_sent_text
                            keep_sentence = True

                            # Discard the original post-sent text
                            if pre_text == '' or sent_text.startswith(pre_text):
                                # the rewritten sentence uses the same pre-text as the original
                                sent_tokens, _ = self.model.tokenize(sent_text[len(pre_text):])
                            else:
                                # tokens_sofar, _ = self.model.tokenize(text_sofar)
                                raise Exception("Rewritten sentence does not start with pre_text. Tokens must be retokenized.")
                            # set no pre_text for the next sentence
                            post_text = ''

                if hallu_state != Hallu.NONE: hallued_sents_text.append(sent_text)

                if keep_sentence:
                    pre_text = post_text

                    fact_sents_text.append(sent_text)
                    fact_sents_tokens.append(sent_tokens)

                    text_sofar = ''.join(fact_sents_text)

                    tokens_sofar = tc.cat([input_tokens]+fact_sents_tokens, dim=-1)
                    decoded_text = self.model.decode(tokens_sofar[0][input_tokens.size(1):])

                    if text_sofar+pre_text != decoded_text:
                        print(f"Decoded text mismatched the test sofar:\nDecoded: {decoded_text}\nText: {text_sofar}")
                        # leng mismatch due to tail text
                        tokens_sofar, _ = self.model.tokenize(text_sofar)
                        pre_text = ''
                        logits = None
                        if past_key_values is not None:
                            cache_seq_len = tc.cat([input_tokens] + fact_sents_tokens[:-1], dim=-1).size(1)
                            past_key_values = past_key_values.crop(cache_seq_len-2)
                    elif nattempt == 1 and hallu_state == Hallu.NONE:
                        logits = logits.post
                    else:
                        if past_key_values is not None:
                            cache_seq_len = tc.cat([input_tokens] + fact_sents_tokens[:-1], dim=-1).size(1)
                            past_key_values = past_key_values.crop(cache_seq_len-1)
                        # discard the logits to enable resample the first token in the next sentence
                        logits = None

                    if nattempt > 1: suppress_EOS = last_fact_sent_suppress_EOS
                    nattempt = 1                
                elif nattempt == self.max_resample_attempts:
                    print("[END] Generation abort due to reaching the max number of resample attempts.")
                    break
                else:
                    print("Discard the current sentence and resample.")
                    nattempt += 1
                    # suppress EOS for next and next next sentences.
                    suppress_EOS = SuppressEOS.ALL

                    if self.resample_policy == 'natural':
                        text_sofar += sent_text
                        tokens_sofar = tc.cat([tokens_sofar, sent_tokens], dim=-1)
                        logits = logits.post
                        pre_text = post_text
                    elif self.resample_policy == 'decode':
                        logits, past_key_values = reset_cache(logits, past_key_values, tokens_sofar)
                        
                    # jump to next iter to avoid end_gen judgement
                    continue

                # early stop generation
                if end_gen:
                    print("[END] Generation complete due to EOS.")
                    break

                rest_new_tokens = max_new_tokens - tokens_sofar.size(1)
                if rest_new_tokens == 0:
                    print(f"[END] Generation abort due to reaching max new tokens: {max_new_tokens}.")
        except Exception as e:
            print(f"[END] Generation abort due to an unexpected error: {e}")
            
        return ''.join(fact_sents_text)

    
    def rewrite_sentence(self, query, sent, props, **args):
        hallu_entities = [prop.entity for prop in props if prop.hallued]
        fact_claims = [prop.fact_claim for prop in props if not prop.hallued]

        fact_claims = []
        for prop in props:
            if prop.hallued: continue
            fact_claim = prop.fact_claim
            if all([hallu_entity not in fact_claim for hallu_entity in hallu_entities]):
                # include only the facts with no conflict with hallucinations
                fact_claims.append(fact_claim)

        if len(fact_claims) == 0:
            return ""
        
        fact_claims = [f"- {fact_claim}" for fact_claim in fact_claims]
        
        user_prompt = '\n'.join([
            f"Query: {query}\n",
            "Fact claims:",
            *fact_claims
        ])
        syst_prompt = rewrite_syst_prompt

        if self.rewrite_cache is None:
            self.rewrite_cache = self.writer.cache_prompt(
                syst_prompt=syst_prompt,
                add_generation_prompt=False
            )

        seq_len = self.rewrite_cache.get_seq_length()
        
        sent = self.writer.complete(
            user_prompt=user_prompt,
            syst_prompt=syst_prompt,
            past_key_values=self.rewrite_cache,
            max_new_tokens=200,
            log=True,
            **args
        )
        
        # discard the kv cache for the variable tokens
        self.rewrite_cache.crop(seq_len)

        sent = sent.strip('"\' \n')
        
        return sent
