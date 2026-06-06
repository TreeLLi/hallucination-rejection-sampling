"""Different prompts (and their logic) used to check for semantic equivalence."""
import os
import logging
from collections import Counter
import numpy as np
import json
import copy

import LLM
import utils
from utils import (
    log_w_indent, cluster_assignment_entropy,
    extract_questions,
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

import addict


fact_extract_syst_prompt = '''You are an information extraction assistant. Your task is to analyze a sentence and output a list of entities and their corresponding fact claims.

---

### Task Instructions

1. Identify all entities in the sentence, excluding the subject.
2. For each entity, generate one fact claim about it based only on the sentence.

---

### Rules

#### Entity Rules:

* Entities include named persons, occupations, organizations, objects, concepts, events, locations, roles, fields, education, times, and numbers.
* Must not include the sentence subject and pronouns like He/She/I/It/This/That/They.
* Copy entities exactly as written in the sentence (no rephrasing, no normalization).


#### Fact Claim Rules:

* Each entity must map to exactly one fact claim.
* A fact claim must:
  - Be a single, complete, grammatically correct sentence.
  - Be fully supported by the input sentence (no added or inferred information).
  - Contain the entity itself explicitly.
* Each fact claim should include only the target entity, unless other entities are strictly necessary to preserve the meaning.


#### Output Format:

* If no entities remain after exclusion, return an empty list.
* Otherwise, return as a list.
* Each line must follow the format: `- entity: fact claim`

---

### Example 1

Input: Lin Li is a Chinese researcher and writer.

Output:
- Chinese: Lin Li is a Chinese.
- researcher: Lin Li is a researcher.
- writer: Lin Li is a writer.

---

### Example 2

Input: Marie Curie won Nobel Prizes in both Physics and Chemistry.

Output:
- Nobel Prizes: Marie Curie won Nobel Prizes.
- Physics: Marie Curie won Nobel Prizes in Physics.
- Chemistry: Marie Curie won Nobel Prizes in Chemistry.

---

### Example 3

Input: He has won several/various/many awards.

Output:
- awards: He has won awards.

'''



question_gen_syst_prompt = '''
You are given a query, a sentence and an entity from that sentence. The sentence is the part of a generated response to the given query. Your task is to generate {n_questions} distinct, natural questions such that:
1. The given sentence serves as a full long-form answer.
2. The given entity serves as a correct short-form answer.

### Instructions:

* Each question must be open-ended (cannot be answered by yes/no).
* Do not mention or hint the answer in the question.
* Each question must be phrased differently (no redundancy, varied structures).

### Output Format:

* output the questions as a list
* each line must follow this format: `- question`

----

### Example 1

#### Input
Query: Tell me about the Eiffel Tower.
Sentence: The Eiffel Tower in Paris was completed in 1889.
Entity: 1889

#### Output
{example1_output}

### Example 2

#### Input
Query: Tell me about Lin Li.
Sentence: Lin Li is a researcher.
Entity: researcher

#### Output
{example2_output}
'''



answer_gen_syst_prompt = '''
You are given from the user a question, along with a query and verified information as context. Answer the question only in plain text. Do not use full sentences. Respond with the fewest words possible, such as a name, place, or thing. If multiple valid answers exist, output up to 5 answers, separated by `;`.
'''



class SpoofData:
    def __getitem__(self, item):
        return f'<{item}>'

    
class UncertaintyEstimator:

    def __init__(self,
                 model,
                 n_questions,
                 n_answers,
                 entailment_type,
                 temperature,
                 top_k,
                 top_p,
                 thinking,
                 device):
        super().__init__()

        self.model = LLM.get_model(model, device)
        self.thinking = thinking
        
        self.n_questions = n_questions
        self.n_answers = n_answers
        self.entailment_type = entailment_type

        self.fact_cache = None
        self.ques_cache = None
        self.answ_cache = None
        
        # prepare question generation prompt
        example1_output = [
            "- In which year was the Eiffel Tower in Paris completed?",
            "- What year was the Eiffel Tower in Paris finished?",
            "- What year marked the completion of the Eiffel Tower in Paris?"
        ]
        example1_output = '\n'.join(example1_output[:n_questions])
        
        example2_output = [
            "- What is Lin Li's profession?",
            "- Which occupation is associated with Lin Li?",
            "- What is Lin Li’s job?",
        ]
        example2_output = '\n'.join(example2_output[:n_questions])
        
        self.question_gen_syst_prompt = question_gen_syst_prompt.format(
            n_questions=n_questions,
            example1_output=example1_output,
            example2_output=example2_output
        )
        
        self.gen_args = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }

        self.qa_gen_args = {
            'temperature': 1,
            'top_k': 50,
            'top_p': 1            
        }


    def get_sent_props(self, sentence, query_entity):
        sentence = sentence.strip()
        props = []

        syst_prompt = fact_extract_syst_prompt

        if self.fact_cache is None:
            self.fact_cache = self.model.cache_prompt(
                syst_prompt=syst_prompt,
                add_generation_prompt=False
            )
            self.fact_seq_len = self.fact_cache.get_seq_length()
        
        output = self.model.complete(
            user_prompt=sentence,
            syst_prompt=syst_prompt,
            past_key_values=self.fact_cache,
            thinking=self.thinking,
            log=True,
            **self.gen_args
        )

        self.fact_cache.crop(self.fact_seq_len)
        
        for line in output.split('\n'):
            line = line.strip()
            if not line.startswith('- ') or len(line.split(':', 1)) < 2:
                log_w_indent(f"Fact decomposition. Invalid proposition line: {line}.")
                continue
            
            entity, claim = line[2:].split(':', 1)
            entity = entity.strip().strip('\'"*')
            entity = utils.find_substring_ignore_case(entity, sentence)
            if entity is None:
                log_w_indent(f"Fact decomposition. Invalid entity in the line: {line}.")
                continue
            
            prop = addict.Dict()
            prop.entity = entity

            if query_entity.lower() in entity.lower():
                log_w_indent(f"Fact decomposition. Invalid subject entity in the line: {line}.")
                continue                
            
            prop.fact_claim = claim.strip()
            props.append(prop)
            
        return props
    
    def get_sent_uncertainty(self, user_query, text_sofar, sentence, query_entity):
        # extract propositations from the sentence
        props = self.get_sent_props(sentence, query_entity)
        
        for prop in props:
            uncertainty, qas = self.get_proposition_uncertainty(user_query, text_sofar, **prop)
            
            prop.uncertainty = uncertainty
            prop.qas = qas

        return props
    
    def base_gen_questions(self, data):
        del data
        raise

    def base_answer_question(self, data):
        del data
        raise

    def base_equivalence(self, data):
        del data
        raise


class SemanticUncertaintyEstimator(UncertaintyEstimator):
    """Questions from context with ground truth answer with. Short answers with context. LLM Entailment without context."""
    def __init__(self, se_w_expected_answer=True, **kwargs):
        super().__init__(**kwargs)

        self.se_w_expected_answer = se_w_expected_answer
        
    def base_gen_questions(self, text_so_far, proposition):
        instruction = f"Generate a list of {self.n_questions} questions, that might have generated the sentence in the context of the preceding original text, as well as their answers. Please do not use specific facts that appear in the follow-up sentence when formulating the question.\nMake the questions and answers diverse. Avoid yes-no questions.\nThe answers should not be a full sentence and as short as possible, e.g. only a name, place, or thing. Use the format \"1. {{question}} -- {{answer}}\""

        if text_so_far == '':
            return f"""You see the sentence:

{proposition}

{instruction}"""
        else:
            return f"""Following this text:

{text_so_far}

You see the sentence:

{proposition}

{instruction}"""
            
        
    def base_answer_question(self, query, text_so_far, question):

        instruction = "Please answer this question in plain text. Do not answer in a full sentence. Answer with as few words as possible, e.g. only a name, place, or thing. When multiple valid answers exist, you should output all valid answers and separate each by ';'."

        if text_so_far == '':
            return f"""We are writing a response to the query "{query}". First, we observe the following question:

{question}

{instruction}"""
        else:
            return f"""We are writing a response to the query "{query}". So far we have written:

{text_so_far}

The next sentence should be the answer to the following question:

{question}

{instruction}"""

    def base_equivalence(self, data):
        prompt = 'Are the following answers equivalent?'
        for i in range(1, self.n_answers + 2):
            prompt += f'\nPossible Answer {i}: ' + '{}'
        prompt += '\nRespond only with "yes" or "no".'

        return prompt.format(
            data['expected_answers'], *data['regen_answers'])
                    
    def get_proposition_uncertainty(self, query, text_so_far, fact_claim, entity):
        data = addict.Dict()
        
        syst_prompt = self.question_gen_syst_prompt
        user_prompt = '\n'.join([
            f"Query: {query}",
            f"Sentence: {fact_claim}",
            f"Entity: {entity}"
        ])

        if self.ques_cache is None:
            self.ques_cache = self.model.cache_prompt(
                syst_prompt=syst_prompt,
                add_generation_prompt=False
            )
            self.ques_seq_len = self.ques_cache.get_seq_length()
        
        output = self.model.complete(
            user_prompt=user_prompt,
            syst_prompt=syst_prompt,
            past_key_values=self.ques_cache,
            thinking=self.thinking,
            log=True,
            **self.gen_args
        )

        self.ques_cache.crop(self.ques_seq_len)

        questions = []
        for question in output.split('\n'):
            question = question.strip()
            if not question.startswith('- '):
                log_w_indent(f"Question Generation. Invalid question line: {question}.")
                continue

            question = question[2:]
            if question in questions:
                log_w_indent(f"Question Generation. Duplicate question: {question}.")
                continue
            questions.append(question)
            
        n_questions = len(questions)
        
        if n_questions == 0:
            raise Exception(f"Failed to generate one question for the prop: {fact_claim}.")
        elif n_questions < self.n_questions:
            log_w_indent(f"Question Generation. Only {n_questions} questions generated.")
        elif n_questions > self.n_questions:
            questions = questions[:self.n_questions]

        expected_answers = [entity for _ in questions]
        
        log_w_indent(f'Extracted questions: {questions}', 2)
        log_w_indent(f'Extracted expected answers: {expected_answers}', 2)

        text_so_far = text_so_far.strip().replace("\n\n", " ")
        
        uncertainties = []
        for qidx, (expected_answer, question) in enumerate(zip(expected_answers, questions)):
            log_w_indent(f'Regenerate answers for question {qidx} "{question}":', 2)

            regen_answers = []
            syst_prompt = answer_gen_syst_prompt
            user_prompt = '\n\n'.join([
                f"Query: {query}",
                f"Verified information: {text_so_far}",
                f"Question: {question}"
            ])
            
            messages = self.model.format_messages(user_prompt=user_prompt, syst_prompt=syst_prompt)
            input_tokens, attention_mask = self.model.tokenize(messages)
            next_token, cache, logits = self.model.next_token(input_tokens,
                                                              attention_mask=attention_mask,
                                                              **self.qa_gen_args)
            seq_len = cache.get_seq_length()

            for i in range(self.n_answers):
                if next_token is None:
                    # the first token of answers can be directly sampled from the cached logits
                    next_token = self.model.sample(input_tokens, logits, **self.qa_gen_args)
                
                _input_tokens = torch.cat([input_tokens, next_token.unsqueeze(-1)], dim=-1)
                _attention_mask = torch.ones_like(_input_tokens, dtype=attention_mask.dtype)
                answer = self.model.complete(input_tokens=_input_tokens,
                                             attention_mask=_attention_mask,
                                             past_key_values=cache,
                                             max_new_tokens=100,
                                             **self.qa_gen_args)
                answer = self.model.decode(next_token) + answer
                next_token = None
                cache.crop(seq_len)
                regen_answers.append(answer)
                
            data[question].expected_answer = expected_answer
            data[question].answers = regen_answers
                
            # << CHECK IF ANSWERS ARE EQUIVALENT >>
            if self.__class__.__name__ in ['QADebertaEntailment', 'QALLMEntailment']:
                
                answers = list(regen_answers)
                if self.se_w_expected_answer:
                    answers.append(expected_answer)
                    
                clusters, uncertainty = self.get_semantic_uncertainty(answers)

                # Account for GPT refusal to answer questions.
                stop_words = ['not available', 'not provided', 'unknown', 'unclear']
                unknown_count = 0
                for answer in answers:
                    for stop_word in stop_words:
                        if stop_word in answer.lower():
                            unknown_count += 1
                            break
                if unknown_count >= len(answers) // 2:
                    logging.warning('Not answerable, setting uncertainty to maximum.')
                    uncertainty = -np.log(1 / len(answers))
                    clusters = str(clusters) + ' not answerable!'

                log_w_indent(f'Semantic Clustering Input: {answers}', 3)
                log_w_indent(f'Semantic Clustering Output: {clusters}, uncertainty: {uncertainty}', 3)
                equiv_response = clusters

                data[question].clusters = clusters
                data[question].uncertainty = uncertainty
            else:
                equiv_prompt = self.base_equivalence({
                    'expected_answers': expected_answer,
                    'regen_answers': regen_answers})
                equiv_response = self.model.complete(user_prompt=equiv_prompt, log=True)
                uncertainty = utils.get_yes_no(equiv_response)

            uncertainties.append(uncertainty)
        
        return np.mean(uncertainties), data

    def is_entailed_largest_cluster(self, expected_answer, answers, clusters):
        # check if expected answer is part of the largest answer cluster
        counter = Counter(clusters)
        largest_cluster = counter.most_common(1)[0][0]

        # select the longest item in the largest cluster
        longest_answer = ''
        for answer, cluster in zip(answers, clusters):
            if cluster != largest_cluster: continue
            if len(answer) < len(longest_answer): continue
            longest_answer = answer

        for cluster_answer in longest_answer.split(';'):
            # this must be 'lax' equivalent entailment.
            if self.are_equivalent(cluster_answer, expected_answer, entailment_type='lax'):
                return True, longest_answer
            
        return False, longest_answer

class QADebertaEntailment(SemanticUncertaintyEstimator):
    """Questions from context with ground truth answer with. Short answers with context. Deberta Entailment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'microsoft/deberta-v2-xlarge-mnli'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entail_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = dict(
            gen_facts=self.gen_facts(data),
            base_gen_questions=self.base_gen_questions(data),
            base_answer_question=self.base_answer_question(data))

        return prompts

    def get_semantic_uncertainty(self, answers):
        semantic_ids = self.get_semantic_ids(answers)
        uncertainty = cluster_assignment_entropy(semantic_ids)
        return semantic_ids, uncertainty

    def check_implication(self, text1, text2):
        inputs = self.tokenizer(text1, text2, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.entail_model(**inputs)
        logits = outputs.logits
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        return largest_index.cpu().item()
    
    def are_equivalent(self, text1, text2, entailment_type=None):
        implication_1 = self.check_implication(text1, text2)
        implication_2 = self.check_implication(text2, text1)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        implications = [implication_1, implication_2]
        entailment_type = entailment_type if entailment_type else self.entailment_type
        if entailment_type == 'lax':
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        elif entailment_type == 'strict':
            semantically_equivalent = (implications[0] == 2) and (implications[1] == 2)
        else:
            raise ValueError
        return semantically_equivalent

    def get_semantic_ids(self, strings_list):
        """Group list of predictions into semantic meaning."""
        # Initialise all ids with -1.
        semantic_set_ids = [-1] * len(strings_list)
        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(strings_list):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id
                for j in range(i + 1, len(strings_list)):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if self.are_equivalent(string1, strings_list[j]):
                        semantic_set_ids[j] = next_id
                next_id += 1
        assert -1 not in semantic_set_ids
        return semantic_set_ids


class QALLMEntailment(QADebertaEntailment):

    def get_all_prompts_for_log(self):
        # Spoof data to log prompting format.
        data = SpoofData()
        prompts = dict(
            gen_facts=self.gen_facts(data),
            base_gen_questions=self.base_gen_questions(data),
            base_answer_question=self.base_answer_question(data),
            base_equivalence=self.base_equivalence(data))

        return prompts

    def base_equivalence(self, data):

        prompt = f"""We are writing an answer to the question "{user_question}"."""

        if data['text_so_far'] is None:
            prompt = prompt + f""" First, we are trying to answer the subquestion "{question}".\n"""
        else:
            prompt = prompt + f""" So far we have written:

{text_so_far}

Next, we are trying to answer the subquestion "{question}".
Does at least one of the following two possible answers entail the other?

Possible Answer 1: {data["text1"]}
Possible Answer 2: {data["text2"]}

Respond with yes or no."""

        return prompt

    def are_equivalent(self, text1, text2, data):

        if text1 == text2:
            log_w_indent(f'Skip entailment check: {text1} == {text2}.', 3)
            return True

        equivalence_prompt = self.base_equivalence({'text1': text1, 'text2': text2, **data})
        equivalence = self.model.complete(equivalence_prompt, 3, data['didx'], EQUIVALENCE, reuse=True)
        uncertainty = utils.get_yes_no(equivalence)

        # If yes in equivalence --> uncertainty == 0 --> return True.
        return {0: True, 1: False}[uncertainty]


def extract_string_between_markers(text, start_marker="```json", end_marker="```"):
    """
    Extracts the string between the specified start and end markers.
    
    Args:
        text (str): The input text containing the markers.
        start_marker (str): The marker indicating the start of the string to extract.
        end_marker (str): The marker indicating the end of the string to extract.
        
    Returns:
        str: The string between the markers, or an empty string if the markers are not found.
    """
    start_index = text.find(start_marker)
    if start_index == -1:
        return text
    
    start_index += len(start_marker)
    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return text
    
    return text[start_index:end_index].strip()


    
ESTIMATORS = dict(
    DeBERTa=QADebertaEntailment,
    LLM=QALLMEntailment,
)

ESTIMATOR_NAMES = sorted(list(ESTIMATORS.keys()))
