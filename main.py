import argparse
import utils
import wandb

import LLM, uncertainty
from generator import UncertaintyGenerator, NaiveGenerator


def main(args):
    model = LLM.get_model(args.model, args.device)

    # diversity regulator
    print(f"Diversity method is {args.diversity_regulator}.")
    if args.diversity_regulator == "temperature":
        args.min_p = None
        args.num_beams = None
        args.num_beam_groups = None
        args.diversity_penalty = None
    elif args.diversity_regulator == "reverse_beam_search":
        args.min_p = None
        args.temperature = None
    elif args.diversity_regulator == "min_p":
        args.num_beams = None
        args.num_beam_groups = None
        args.diversity_penalty = None
    else:
        raise KeyError("diversity_regulator should be one of 'min_p', 'reverse_beam_search', or 'temperature'.")


    if args.generator == 'uncertainty':
        generator = UncertaintyGenerator(
            model,
            uncertainty_estimator=args.uncertainty_estimator,
            uncertainty_base_model=args.uncertainty_base_model,
            uncertainty_model_thinking=args.uncertainty_model_thinking,
            n_questions=args.n_questions,
            n_answers=args.n_answers,
            entailment_type=args.entailment_type,
            uncertainty_threshold=args.uncertainty_threshold,
            max_resample_attempts=args.max_resample_attempts,
            max_repetition_attempts=args.max_repetition_attempts,
            rewrite=args.rewrite,
            resample_policy=args.resample_policy,
            entail_check=args.entail_check,
            device=args.device,
            se_w_expected_answer=args.se_w_expected_answer,
            temperatures=args.temperatures,
            top_k=args.top_k,
            top_p=args.top_p
        )
    else:
        generator = NaiveGenerator(model, device=args.device)
        
    syst_prompt = "You are given a query from the user asking for information. Write the answer in English characters. Output plain text only. Do not use formatting styles, symbols, or headings. Each sentence should provide different information and must not repeat the same content. Output only the final answer to the query. Do not generate explanations, reasoning, or commentary for the answer. Do not ask follow-up questions."
    
    if args.new_words is not None:
        if args.new_words[0] == '<':
            args.new_words = int(args.new_words[1:])
            syst_prompt += f" The response must be no more than {args.new_words} words in total."
        else:
            syst_prompt += f" The response must be around {args.new_words} words in total."
        
    wandb.config.main_syst_prompt = syst_prompt
    
    with open(f"{args.data_dir}/{args.data}.txt", 'r') as f:
        entities = f.readlines()
    
    columns = ["query", "generation"]
    table = wandb.Table(columns=columns)
    
    output = []
    for entity in entities:
        entity = entity.strip()
        
        if 'factscore' in args.data or args.data == 'longfact_description':
            query = f"Tell me about {entity}."
        else:
            query = entity

        if query.endswith('..'): query = query[:-1]
        
        print('--------------------------------------------------------------------------------')
        print(f"Query:\t{query}")
        
        output.append(f'Query:\t{query}')
        generation = generator.generate(
            user_prompt=query,
            syst_prompt=syst_prompt,
            entity=entity,
            thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            temperatures=args.temperatures,
            top_k=args.top_k,
            top_p=args.top_p,
            diversity_penalty=args.diversity_penalty,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            min_p=args.min_p,
        )

        if utils.is_non_answer(generation):
            print(f"Generation is determined as non-answered: {generation}")
            generation = ""
        
        output.append(f'Response: {generation}')
        print(f'Response:\t{generation}')
        
        table.add_data(query, generation)

        break
        
    wandb.log({"generation": table})
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-id", type=str, default='SEGen',
        help="the project ID in wandb.")
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument(
        "--device", choices=['cuda', 'cpu'], default='cuda',
        help="the compute device to run the model.")
    parser.add_argument(
        "--data-dir", type=str, default='data',
        help="dir to find the data")
    parser.add_argument(
        "-d", "--data",
        choices=['factscore', 'longfact_description', 'custom'],
        default=['custom'],
        help="the name of persons to generate, or the name of the data file with the person names.")
    parser.add_argument(
        "-m", "--model", choices=LLM.MODEL_NAMES, default='qwen3-0.6b',
        help="base LLM model to respond user prompt.")
    parser.add_argument(
        "-g", "--generator", choices=['naive', 'uncertainty'], default='uncertainty',
        help="the generation wrapper method to use.")
    parser.add_argument(
        "--thinking", action='store_true', default=False,
        help="enable thinking mode.")    
    
    parser.add_argument(
        "--uncertainty-estimator", choices=uncertainty.ESTIMATOR_NAMES, default='DeBERTa',
        help="the uncertainty estimator to use.")
    parser.add_argument(
        "-um", "--uncertainty-base-model", choices=LLM.MODEL_NAMES, default=None,
        help="base LLM model to compute uncertainty.")
    parser.add_argument(
        "-umt", "--uncertainty-model-thinking", action='store_true', default=False,
        help="enable thinking mode for uncertainty model.")        
    parser.add_argument(
        "--entailment_type", type=str, default='lax',  # or strict
        help="Lax or strict entailment.")
    parser.add_argument(
        "--n-questions", type=int, default=3,
        help="Number of questions to ask per proposition.")
    parser.add_argument(
        "--n-answers", type=int, default=3,
        help="Number of answers per question.")
    parser.add_argument(
        "--max-resample-attempts", type=int, default=10,
        help="max number of attempts to resample a sentence when hallucinated.")
    parser.add_argument(
        "--max-repetition-attempts", type=int, default=10,
        help="max number of attempts to resample a sentence when repeated.")
    parser.add_argument(
        "--uncertainty-threshold", type=float, default=0.3,
        help="the threshold of uncertainty for a proposition to be determined as hallucinated.")
    parser.add_argument(
        "-ea", "--se-expected-answer", dest="se_w_expected_answer", action='store_true', default=False,
        help="Add expected answer to SE generated answers.")
    parser.add_argument(
        "--rewrite", action='store_true', default=False,
        help="Enable rewriting if a sentence is partially hallucinated.")
    parser.add_argument(
        "--resample-policy",
        choices=['decode', 'natural'],
        default='decode',
        help="decode: randomply resampling; natural: keep and sample next")
    
    parser.add_argument(
        "--entail-check", action='store_true', default=False,
        help="check if the factual claim entailed by the largest cluster answer")
    
    parser.add_argument(
        "-w", "--new-words", type=str, default=None,
        help="number of words for response specified in the prompt. '<' for no more than")
    parser.add_argument(
        "--max-new-tokens", type=int, default=5000,
        help="max number of new tokens to be generated.")
    parser.add_argument("--diversity-regulator", 
        choices=["temperature", "reverse_beam_search", "min_p"], default="temperature",
        help="this is a helper flag to make sure multiple diversity settings are not used at once. note that the 'min_p' setting still uses temperature, while reverse beam search does not.")
    parser.add_argument(
        "-t", "--temperatures", nargs='+', type=float,
        default=[0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9],
        help="temperature for LLM decoding.")
    parser.add_argument(
        "-k", "--top-k", type=int, default=50,
        help="temperature for LLM decoding.")
    parser.add_argument(
        "-p", "--top-p", type=float, default=1,
        help="temperature for LLM decoding.")
    parser.add_argument(
        "--diversity-penalty", type=float, default=None,
        help="number of beams for diverse beam search")
    parser.add_argument(
        "--num-beams", type=int, default=None,
        help="number of beams for diverse beam search")
    parser.add_argument(
        "--num-beam-groups", type=int, default=None,
        help="number of beam groups for diverse beam search")
    parser.add_argument(
        "--min-p", type=float, default=None,
        help="min p for diversity regulation")
        
    args = parser.parse_args()
    if not args.uncertainty_base_model:
        args.uncertainty_base_model = args.model

    wandb.login()
    run = wandb.init(project=args.project_id, config=args, job_type='inference')
    
    log_path = f'log/{run.name}.txt'
    utils.setup_logger(log_path)
    
    wandb.save(log_path, policy="live")
    
    main(args)
