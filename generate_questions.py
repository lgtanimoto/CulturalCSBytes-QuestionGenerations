# built-in python modules
import json
import sys
import os
import warnings
import random
from typing import Dict, List
import datetime

# langchain
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector

# python libraries
from dotenv import load_dotenv
from tqdm import tqdm

FILE_PREF = "object_data\\"


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(filename):
    with open(FILE_PREF + filename, 'r') as f:
        data = json.load(f)
    return data


lo_list = read_json("learning_objectives.json")
lo_code_groups = read_json("lo_code_groups.json")
interest_to_area = read_json("topics_to_interest_areas_map.json")


def group_examples(examples) -> Dict[str, List]:
    # categorize examples by learning objective group
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


def make_prompt(examples, num_examples: int, coding: bool = False) -> FewShotPromptTemplate:
    ex_groups = group_examples(examples)

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
        exs = random.sample(examples, min(len(examples), 3))
        few_shot_prompt = FewShotPromptTemplate(
            examples=exs,
            example_prompt=example_prompt,
            prefix="""System Message: You are a high school computer science teacher who wants to connect CS concepts to the diverse interests of your students so that they can better see themselves working in CS. Applications of CS to different fields of study often have similar solutions in code and teach the same underlying CS principles. And by seeing questions related to a wide variety of disciplines, students can better appreciate that CS is everywhere, for everyone, and that the essence of good computer science is to find the common patterns in problems from all aspects of life and to develop solutions for them.\n\nTask: Given a CS learning objective and a topic of interest to your students, generate a question in JSON format that applies the learning objective to a scenario related to the provided topic. The question should involve a code snippet specified in the quorum programming language as shown in the example questions.""",    
            suffix="Learning objective: {learning_objective}\nTopic: {topic}\nQuestion:",
            input_variables=["learning_objective", "topic"]
        )
    else:
        example_selector = CustomExampleSelector(examples)
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="""System Message: You are a high school computer science teacher who wants to connect CS concepts to the diverse interests of your students so that they can better see themselves working in CS. Applications of CS to different fields of study often have similar solutions in code and teach the same underlying CS principles. And by seeing questions related to a wide variety of disciplines, students can better appreciate that CS is everywhere, for everyone, and that the essence of good computer science is to find the common patterns in problems from all aspects of life and to develop solutions for them.\n\nTask: Given a CS learning objective and a topic of interest to your students, generate a question in JSON format that applies the learning objective to a scenario related to the provided topic. Specify the content of any charts or graphics referenced in your question.""",    
            suffix="Learning objective: {learning_objective}\nTopic: {topic}\nQuestion:",
            input_variables=["learning_objective", "topic"]
        )

    return few_shot_prompt


def generate_qs(llm, topic_list, examples: list[dict], num_examples: int, output_folder: str, coding: bool = False, verbose: bool = False) -> list[dict]:
    generated_qs = []
    for topic in tqdm(topic_list, desc="Question Generation Progress"):
        # randomly select learning standard
        lo_code, obj = random.choice(list(lo_list.items()))
        
        # create prompt
        prompt = make_prompt(examples, num_examples, coding).format(learning_objective=obj, topic=topic)

        # get gpt-4 response
        res = llm(prompt)
        q_str = "{" + res.split("{")[-1].split("}")[0] + "}"
        q = {
            'MQCode': lo_code,
            'learning_objective': obj,
            'interest_area': interest_to_area[topic],
            'topic': topic,
            'coding': coding,
            'question_str': q_str,
        }
        if verbose:
            print("\n" + q_str + "\n")
        
        generated_qs.append(q)

        # output question json to file
        f_name = q['MQCode'] + "-" + q['topic'].replace(" ", "_")
        write_json(q, output_folder + f_name)

    return generated_qs


def generate_topic_list(batch_size: int, interest_areas: list):
    interests = read_json("interests.json")
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


def usage():
    print('''usage: python generate_questions.py <batch_size> [options]

required arguments:
  batch_size                The number of questions to generate.

optional arguments:
  -h, --help                Show this help message and exit.
  -l, --llm                 The name of the LLM to use for generating questions. Defaults to 'gpt-4'.
  -e, --examples-file       The relative path to the file containing a list of few-shot examples.
                            Defaults to 'object_data\\coding_question_examples.json' if '--coding' flag is included.
                            Defaults to 'object_data\\default_question_examples.json' if '--coding' flag is ommitted.
  -n, --num-examples        The number of examples to use in few-shot prompting. Defaults to 3.
  -o, --output-folder       The relative path to the folder where generated questions should be written.
                            Defaults to "questions\\date\\time\\".
  -c, --coding              Indicates generated questions should include code. Defaults to false by ommission.
  -i, --interest-areas      A comma-separated list of interest areas to generate questions for. Defaults to all interest areas.
                            Notes: interest areas are case-sensitive and multi-word interest areas should use dashes.
                            e.g. '--interest-areas=Diversity-and-inclusion,Business'.
  -a, --api-key             Your OpenAI-api-key. Defaults to reading from .env file.
  -v, --verbose             Enable verbose mode to print generated questions.\n''')


def get_opt_match(target: str):
    opts = sys.argv[2:]
    opt_match = [opt for opt in opts if opt.startswith(target)]
    if len(opt_match) != 0:
        return opt_match[0].split("=")[-1]
    elif target[1:3] in opts:
        return opts[opts.index(target[1:3]) + 1]
    else:
        return None


def main():
    if len(sys.argv) < 2 or "--help" == sys.argv[1] or "-h" == sys.argv[1]:
        usage()
        exit(1)

    batch_size = int(sys.argv[1])  # number of questions to generate
    options = sys.argv[2:]

    """parse OpenAI api key"""
    openai_api_key = get_opt_match("--api-key")
    if openai_api_key is None:
        load_dotenv()
        openai_api_key = os.getenv("openai_api_key")

    """parse llm name"""
    model_name = get_opt_match("--llm")
    if model_name is None:
        model_name = "gpt-4"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key)
    except:
        print("Invalid LLM name.\n")
        usage()
        exit(1)

    """check for coding flag"""
    coding = "--coding" in options or "-c" in options
    
    """parse path to examples file"""
    path_to_examples = get_opt_match("--examples-file")
    if path_to_examples is None:
        if coding:
            path_to_examples = FILE_PREF + "coding_question_examples.json"
        else:
            path_to_examples = FILE_PREF + "default_question_examples.json"
    try:
        with open(path_to_examples, 'r') as ex_f:
            examples = json.load(ex_f)
    except:
        print("Invalid examples file.\n")
        usage()
        exit(1)
    
    """parse number of examples to provide for prompting"""
    num_examples = get_opt_match("--num-examples")
    if num_examples is not None:
        num_examples = int(num_examples)
    else:
        num_examples = 3

    """parse destination path of where to save generated questions"""
    output_folder = get_opt_match("--output-folder")
    if output_folder is None:
        dt = str(datetime.datetime.now()).replace(" ", "\\").split(".")[0].replace(":","-")
        output_folder = "questions\\" + dt + "\\"
    # create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    """parse interest area(s) to generate questions for"""
    interest_areas = get_opt_match("--interest-areas")
    if interest_areas is not None:
        interest_areas = interest_areas.replace("-", " ").split(",")
    else:
        interest_areas = read_json("interest_areas.json")

    topic_list = generate_topic_list(batch_size, interest_areas)

    # set verbose option
    verbose = "--verbose" in options or "-v" in options

    generate_qs(llm, topic_list, examples, num_examples, output_folder, coding, verbose)


if __name__ == "__main__":
    main()
    exit(0)
