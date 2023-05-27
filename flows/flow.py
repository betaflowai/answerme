"""
build a flow for QA via LLM and RAG
ref
https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
"""
import argparse
import logging
import sys
from typing import List
import yaml
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from tqdm import tqdm


def write_qa(header: str, out_file_path: str, questions: List[str], answers: List[str]) -> None:
    out_file = open(out_file_path, "w")
    out_file.write(f'#\n{header}\n#\n')
    out_file.write("\n")
    assert len(questions) == len(answers)
    N = len(questions)
    for i in range(N):
        out_file.write(f"Q:{questions[i].strip()}")
        out_file.write("\n")
        out_file.write(f"A:{answers[i].strip()}")
        out_file.write("\n")
        out_file.write("---")
        out_file.write("\n")
    out_file.close()


def get_hugging_face_llm_interface(config: dict, llm_name_id_map):
    model_name = config['model']['name']
    repo_id = llm_name_id_map.get(model_name, None)
    if not repo_id:
        raise ValueError(f"Invalid model name {model_name}")
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 64})
    return llm


def get_openai_llm_interface(config: dict):
    model_name = config['model']['name']
    if model_name == 'default':
        return OpenAI()
    # model_name for OpenAI
    # https://platform.openai.com/docs/models/gpt-3
    model_id = config['model_map'][model_name]
    return OpenAI(model_name=model_id)


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


def get_docs_retriever(config: dict, context_doc: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(context_doc)

    embeddings = OpenAIEmbeddings()  # fixme, decide it from embeddings
    docs_retriever = Chroma.from_texts(texts, embeddings,
                                       metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    return docs_retriever


def run_faq_pipeline(llm, docs_retriever, questions):
    logger_ = logging.getLogger()
    logger_.info('Starting faq pipeline')
    answers = []
    for question in tqdm(questions, desc='answering questions'):
        docs = docs_retriever.get_relevant_documents(question)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question).strip()
        answers.append(answer)
    assert len(questions) == len(answers)
    return answers

    # write answers


def run_master_pipeline(config: dict, context_doc: str, questions: List[str]) -> List[str]:
    """

    :param config:
    :param questions:
    :return:
    """
    logger.info(f'Getting LLM')
    if config['model']['provider'] == 'huggingface':
        llm = get_hugging_face_llm_interface(config=config, llm_name_id_map=config['model_map'])
    elif config['model']['provider'] == 'openai':
        llm = get_openai_llm_interface(config=config)
    else:
        raise ValueError("Invalid provider")
    logger.info(f'Specifying task')
    if config['task'] == 'faq':
        docs_retriever = get_docs_retriever(config, context_doc)
        answers = run_faq_pipeline(llm, docs_retriever, questions)
        return answers


    else:
        raise ValueError(f"unsupported task {config['task']}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    #
    parser = get_parser()
    args = parser.parse_args()
    #
    with open(args.flow) as stream:
        config = yaml.safe_load(stream)
    # get questions
    q_file = open(args.q_file, "r")
    questions = q_file.readlines()
    context_doc_file = open(args.docs, "r")
    context_doc = context_doc_file.read()
    logger.info(f'Staring master pipeline')
    answers = run_master_pipeline(config=config, context_doc=context_doc, questions=questions)
    write_qa(out_file_path=args.qa_file, header="", questions=questions, answers=answers)
    logger.info(f'Finished writing resulted QA to {args.qa_file}')
    # FIXME , why stuck at the end, must end manually
    """
    Traceback (most recent call last):
      File "/home/mbaddar/mbaddar/llm_project/llm-flow/venv310/lib/python3.10/site-packages/posthog/client.py", line 400, in join
        consumer.join()
      File "/usr/lib/python3.10/threading.py", line 1096, in join
        self._wait_for_tstate_lock()
      File "/usr/lib/python3.10/threading.py", line 1116, in _wait_for_tstate_lock
        if lock.acquire(block, timeout):
    KeyboardInterrupt: 
    
    Process finished with exit code 0
    """
