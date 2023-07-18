# built-in python modules
import json
import sys
import os
import warnings
import random
from typing import Dict, List
import datetime
import argparse
from typing import Tuple

# langchain
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector

# python libraries
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken


FILE_PREF = "object_data\\"


"""Writes json file to filepath. Ommit .json extension in function call"""
def write_json(data, filename):
    with open(filename + ".json", 'w') as f:
        json.dump(data, f, indent=4)


"""Reads json file in object_data directory"""
def read_json(filename):
    with open(FILE_PREF + filename + ".json", 'r') as f:
        data = json.load(f)
    return data


"""declare global variables"""
lo_list = read_json("learning_objectives")
lo_list_coding = read_json("learning_objectives_coding")
lo_code_groups = read_json("lo_code_groups")
context_windows = read_json("llm_context_windows")
interests = read_json("interests")
question_schema = read_json("question_schema")


"""Stores and returns examples by learning objective group"""
def group_examples(examples) -> Dict[str, List]:
    ex_groups = {
        'C': [],
        'N': [],
        'D': [],
        'A': [],
        'I': [],
    }

    for ex in examples:
        ex_groups[ex['MQCode'][1]].append(ex)

    return ex_groups


"""Generates prompt template for few-shot learning"""
def make_prompt(
    examples,
    num_examples: int,
    coding: bool = False
) -> FewShotPromptTemplate:

    ex_groups = group_examples(examples)

    """Custom example selector to select examples based on input variables"""
    class CustomExampleSelector(BaseExampleSelector):
        def __init__(self, examples: List[Dict[str, str]]):
            self.examples = examples

        def add_example(self, example: Dict[str, str]) -> None:
            # Add new example to store for a key.
            self.examples.append(example)

        def select_examples(self, input_variables: Dict[str, str]):
            # Select which examples to use based on the inputs.
            for code, obj in lo_list.items():
                if obj == input_variables['learning_objective']:
                    similar_exs = ex_groups[code[1]]
                    few_exs = random.sample(similar_exs, min(len(similar_exs), num_examples))
                    return few_exs

    question_template = """Learning objective: {learning_objective}\nTopic: {topic}\nQuestion:\n```json\n{{{question_str}}}\n```"""

    example_prompt = PromptTemplate(
        input_variables=["learning_objective", "topic", "question_str"],
        template=question_template,
    )

    if coding:
        """Specialize prompt for coding questions"""
        with open("object_data\\system_message_coding.txt", 'r') as f:
            prefix = f.read()
        exs = random.sample(examples, min(len(examples), num_examples))
        few_shot_prompt = FewShotPromptTemplate(
            examples=exs,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix="Learning objective: {learning_objective}\nTopic: {topic}\nQuestion:",
            input_variables=["learning_objective", "topic"]
        )
    else:
        """Specialize prompt for non-coding questions"""
        with open("object_data\\system_message_default.txt", 'r') as f:
            prefix = f.read()
        example_selector = CustomExampleSelector(examples)
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix="Learning objective: {learning_objective}\nTopic: {topic}\nQuestion:",
            input_variables=["learning_objective", "topic"]
        )

    return few_shot_prompt


"""Creates mapping from subtopic to interest area"""
def generate_topic_to_area_map():
    topics_to_interest_areas = {
        subtopic: area
        for (area, subtopics) in list(interests.items())
        for subtopic in subtopics
    }
    return topics_to_interest_areas


"""Selects model based on window size and prompt length"""
def get_llm(prompt: str, model_name: str, openai_api_key: str):
    if model_name not in context_windows.keys():
        print("Invalid LLM name. Valid llm options are as follows:")
        print(*([" "] + list(read_json("llm_context_windows").keys())), sep='\n > ')
        print()
        exit(1)

    num_tokens = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    llms_2049 = [model for model, window in context_windows.items() if window == 2049]

    if num_tokens + 500 > context_windows[model_name]:
        if model_name == "gpt-4":
            model_name = "gpt-4-32k"
        elif model_name == "gpt-3.5-turbo" or model_name == "text-davinci-003":
            model_name = "gpt-3.5-turbo-16k"
        elif model_name in llms_2049:
            model_name = "text-davinci-003"
        else:
            print("Context window error: Consider using gpt-4-32k or decreasing the number of prompt examples.")
            exit(1)

        # check that window size of new model is sufficient with recursive call
        return get_llm(prompt, model_name, openai_api_key)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key) # type: ignore

    return llm


"""Verifies that the generated question schema is valid"""
def verify_generation(q_str: str, coding: bool):
    try:
        if not coding:
            return question_schema[:-1].sort() == list(json.loads(q_str).keys()).sort()
        else:
            return question_schema.sort() == list(json.loads(q_str).keys()).sort() and json.loads(q_str)['code_snippet'] != ""
    except json.decoder.JSONDecodeError:
        return False


"""Randomly selects a learning objective based on the given parameters"""
def get_learning_objective(coding: bool, filter_grades: bool) -> Tuple[str, str]:
    los = lo_list_coding if coding else lo_list
    if filter_grades:
        los = {key: value for (key, value) in list(los.items()) if key[0] != '5'}
    return random.choice(list(los.items()))


"""Generates a batch of questions using the LLM, the given examples, and a list of topics.
Randomizes the learning standards used to generate the questions."""
def generate_questions(
    llm,
    topic_list: list[str],
    examples: list[dict],
    num_examples: int,
    output_folder=None,
    coding: bool = False,
    verbose: bool = False,
    filter_grades: bool = False,
    debug: bool = False,
) -> list[dict]:

    generated_qs = []
    interest_to_area = generate_topic_to_area_map()

    for topic in tqdm(topic_list, desc="Question Generation Progress"):
        # randomly select learning standard
        lo_code, obj = get_learning_objective(coding, filter_grades)

        # create prompt
        prompt = make_prompt(
            examples,
            num_examples,
            coding
        ).format(
            learning_objective=obj,
            topic=topic
        )

        if debug:
            print("\n\n" + prompt)
        
        # get gpt-4 response
        res = llm(prompt)
        q_str = "{" + res.split("{")[-1].split("}")[0] + "}"

        if debug:
            print(q_str + "\n")

        # check that generated question schema is valid
        if not verify_generation(q_str, coding):
            print("\nGeneration Error: Invalid question schema\n")
            continue

        q = {
            'MQCode': lo_code,
            'learning_objective': obj,
            'interest_area': interest_to_area[topic],
            'topic': topic,
            'coding': coding,
            'question_str': q_str,
        }
        generated_qs.append(q)

        """Save / print question"""
        if output_folder:
            f_name = q['MQCode'] + "-" + q['topic'].replace(" ", "_").replace("/", "-")
            write_json(q, output_folder + f_name)
            if verbose:
                print(f"\nQuestion saved at {output_folder + f_name}")
        else:
            print(q_str)

    return generated_qs


"""Generates a list of topics to be used for generating questions"""
def generate_topic_list(batch_size: int, interest_areas: list):
    curr_interests = []
    # get unique list of topics
    for area in interest_areas:
        for interest in interests[area]:
            curr_interests.append(interest)

    # sample batch_size topics from unique list
    topic_list = []
    while batch_size > len(curr_interests):
        for interest in curr_interests:
            topic_list.append(interest)
        batch_size -= len(curr_interests)
    for interest in random.sample(curr_interests, batch_size):
        topic_list.append(interest)

    return topic_list


def main():
    # get CLI arguments
    parser = argparse.ArgumentParser(description="Generate questions using OpenAI's API.")

    """define CLI arguments"""
    parser.add_argument("num_questions", type=int, help="The number of questions to generate.")
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo", help="The name of the LLM to use for generating questions. Defaults to 'gpt-4'. Other model options include: 'gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'text-curie-001'. Note, the gpt-4 and gpt-3.5 options auto-fit their context window to the provided prompt length.")
    parser.add_argument("-e", "--examples-file", type=str, default=None, help="The relative path to the .json file containing a list of few-shot examples. Defaults to 'object_data\\coding_question_examples.json' if '--coding' flag is included. Defaults to 'object_data\\default_question_examples.json' if '--coding' flag is ommitted.")
    parser.add_argument("-n", "--num-examples", type=int, default=3, help="The number of examples to use in few-shot prompting. Defaults to 3.")
    parser.add_argument("-s", "--save", action="store_true", help="Include flag to save generated questions to .json files. Specify output folder with '-o, --output-folder'.")
    parser.add_argument("-o", "--output-folder", type=str, default=None, help="The relative path to the folder where generated questions should be saved. Defaults to 'questions\\date\\time\\' if '-s, --save' flag is included.")
    parser.add_argument("-i", "--interest-areas", type=str, nargs="+", default=read_json("interest_areas"), help="The interest areas to generate questions for. Defaults to all interest areas. Notes: interest areas are case-sensitive and multi-word interest areas should use dashes. e.g. '--interest-areas=Diversity-and-inclusion Business'. Refer to object_data\\interest_areas.json for a list of valid interest areas. Interest areas should be separated by spaces.")
    parser.add_argument("-a", "--api-key", type=str, default=None, help="Your OpenAI-api-key. Defaults to reading from .env file.")
    parser.add_argument("-c", "--coding", action="store_true", help="Include flag to generate questions with code. By default, questions do not include code.")
    parser.add_argument("-f", "--filter-grades", action="store_true", help="Include flag to filter out 11th and 12th grade learning standards when generating questions. By default, all learning standards are used during generation.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode to print generated questions.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode to print debug messages.")

    """parse CLI arguments"""
    args = parser.parse_args()
    num_examples = args.num_examples
    model_name = args.model
    batch_size = args.num_questions
    interest_areas = args.interest_areas
    filter_grades = args.filter_grades
    coding = args.coding
    verbose = args.verbose
    output_folder = args.output_folder
    save = args.save
    debug = args.debug

    """get OpenAI API key"""
    if not args.api_key:
        load_dotenv()
        openai_api_key = str(os.getenv("OPENAI_API_KEY"))
    else:
        openai_api_key = args.api_key
        # save argument api-key to .env file
        if not os.path.exists(".env"):
            with open(".env", 'w') as f:
                f.write("OPENAI_API_KEY=" + openai_api_key)
    if not openai_api_key:
        print("Please provide an OpenAI API key.")
        exit(1)

    """generate path to examples file"""
    path_to_examples = args.examples_file
    if path_to_examples is None:
        if args.coding:
            path_to_examples = FILE_PREF + "coding_question_examples"
        else:
            path_to_examples = FILE_PREF + "default_question_examples"
    try:
        with open(path_to_examples.split(".json")[0] + ".json", 'r') as ex_f:
            examples = json.load(ex_f)
    except FileNotFoundError:
        print(f"Error: unable to open {path_to_examples}.json")
        exit(1)

    """parse output folder"""
    if save and not output_folder:
        dt = str(datetime.datetime.now()).replace(" ", "\\").split(".")[0].replace(":", "-")
        output_folder = "questions\\" + dt + "\\"
    if output_folder:
        # add trailing slash to output folder if not present
        output_folder += "\\" if output_folder[-1] != "\\" else ""
        # create the output folder if it doesn't exist
        if batch_size > 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)

    topic_list = generate_topic_list(batch_size, interest_areas)

    # instantiate the llm
    ex_prompt = make_prompt(examples, num_examples, coding).format(learning_objective=lo_list["1A08"], topic="bar")
    llm = get_llm(ex_prompt, model_name, openai_api_key)

    if verbose:
        print(f"\nllm = {llm._identifying_params['model_name']}\nbatch_size = {batch_size}\napi_key = {openai_api_key}\nexamples_file = {path_to_examples}\nnum_examples = {num_examples}\noutput_folder = {output_folder}\ncoding = {coding}\ninterest_areas = {interest_areas}\nfilter_grades = {filter_grades}\n")

    generate_questions(llm, topic_list, examples, num_examples, output_folder, coding, verbose, filter_grades, debug)


if __name__ == "__main__":
    main()
    exit(0)
