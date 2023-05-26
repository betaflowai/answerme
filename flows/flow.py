# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token
import argparse
import logging
from langchain.document_loaders import TextLoader
import yaml
from tqdm import tqdm
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain


def get_hugging_face_llm_interface(config: dict, llm_name_id_map):
    model_name = config['model']['name']
    repo_id = llm_name_id_map.get(model_name, None)
    if not repo_id:
        raise ValueError(f"Invalid model name {model_name}")
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64})
    return llm


def get_openai_llm_interface(config: dict):
    model_name = config['model']['name']
    if model_name == 'default':
        return OpenAI()
    # model_name for OpenAI
    # https://platform.openai.com/docs/models/gpt-3
    model_id = config['model_map'][model_name]
    return OpenAI(model_name=model_id)


"""
refs
https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q-file', type=str, required=True,
                        help='Input questions file. One question per line')
    parser.add_argument('--flow', type=str, required=True,
                        help='flow.yaml file')
    parser.add_argument('--qa-file', type=str, required=True,
                        help='Output file with QA')
    parser.add_argument('--docs', type=str, required=True,
                        help='domain docs')
    return parser


def run_pipeline(config: dict, args):
    llm = None
    if config['model']['provider'] == 'huggingface':
        llm = get_hugging_face_llm_interface(config=config, llm_name_id_map=config['model_map'])
    elif config['model']['provider'] == 'openai':
        llm = get_openai_llm_interface(config=config)
    else:
        raise ValueError("Invalid provider")
    loader = TextLoader(args.docs, encoding='utf8')
    documents = loader.load()
    chain = load_qa_chain(llm, chain_type="stuff")
    answers = []
    in_file = open(args.q_file, "r")
    questions = in_file.readlines()
    in_file.close()
    # answer questions
    N = len(questions)
    for question in tqdm(questions, desc='answering questions'):
        question = question.strip()
        if config['prediction'] == 'chain':
            answer = chain.run(input_documents=documents, question=question).strip()
        elif config['prediction'] == 'llm':
            answer = llm.predict(question).strip()
        else:
            raise ValueError('Unknown prediction methods')
        # answer = llm.predict(question).strip()
        answers.append(answer)
    assert len(questions) == len(answers)
    # write answers
    with open(args.qa_file, "w") as fo:
        for i in range(N):
            str_to_write = f"Q:{questions[i].strip()}\nA:{answers[i].strip()}\n---\n"
            fo.write(str_to_write)
        fo.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    #
    parser = get_parser()
    args = parser.parse_args()
    #
    with open(args.flow) as stream:
        config = yaml.safe_load(stream)
    run_pipeline(config, args)
