from template import *
from basic import *
from tqdm import tqdm
from termcolor import colored
import llama_factory_inference
import os


def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


model, template_version = "vicuna", "D_F"
clear()
while True:
    print("AutoRE Loaded Done")
    sentence = input("input a document:")
    print(colored(f"Document: {sentence}\n", 'yellow'))
    fact_list_prompt = templates[template_version]["fact_list_template"].format(
        sentences=sentence)
    print(colored(f'fact_list_prompt:\n{fact_list_prompt}\n', 'green'))
    ori_fact_list = llama_factory_inference.llama_factory_inference(
        model, fact_list_prompt)
    facts = get_fixed_facts(ori_fact_list, sentence)
    print(colored(f'    Extracted Facts:{facts}\n', 'blue'))

# The president has declared an emergency in the state of Utah due to the civic unrest caused by weird people.
