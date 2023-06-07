"""
https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
https://docs.langchain.com/docs/components/chains/index_related_chains
"""
import argparse
import logging
import os
import zipfile

import PyPDF2
from termcolor import colored
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def convert_pdf_to_text(pdf_file_obj):
    pdfreader = PyPDF2.PdfReader(pdf_file_obj)
    n = len(pdfreader.pages)
    all_text = ""
    for i in range(n):
        page = pdfreader.pages[i]
        all_text += f"## Page {i + 1} ##\n\n===\n\n"
        all_text += page.extract_text()
        all_text += "\n\n====\n\n"
    return all_text

def get_context_text(zip_file_path):
    logger = logging.getLogger()
    context_text = ""
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if file_name.endswith('.txt'):
                logger.info(f'Appending text from {os.path.join(zip_file_path, file_name)} to context string')
                text = zip_file.read(file_name).decode()
                context_text += text
                context_text += "\n---\n"
            elif file_name.endswith('.pdf'):
                logger.info(f'Appending text from {os.path.join(zip_file_path, file_name)} to context string')
                # TODO
                #   https://stackoverflow.com/questions/62055857/reading-a-pdf-from-a-zipfile
            else:
                logger.warning(f'file {file_name} is not a text file, and I am not processing it')
    return context_text


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-zip-path', type=str, required=True, help='Root for context zip file')
    return parser


if __name__ == '__main__':
    colorama_init()
    logging.basicConfig(level=logging.CRITICAL)  # hack to disable logging
    logger = logging.getLogger()
    chunk_size = 300
    chunk_overlap = 20
    #
    parser = get_parser()
    args = parser.parse_args()
    # build chain
    print("Loading LLM Model ###")
    model_name = "text-davinci-003"
    temperature = 0.2
    llm = OpenAI(model_name=model_name, temperature=temperature)
    context_text = get_context_text(zip_file_path=args.context_zip_path)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(context_text)
    chain = load_qa_chain(llm, chain_type="stuff")
    embeddings = OpenAIEmbeddings()
    print('Indexing context docs')
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    while True:
        # print(colored('Input question', 'red'))
        print(f"{Fore.GREEN}Input question : {Style.RESET_ALL}!")
        question = input()
        # Generic LLM
        answer1 = llm.predict(question)
        print(f'{Fore.BLUE}Answer with Generic LLM:\n{answer1.strip()}\n==={Style.RESET_ALL}')
        # Context-Aware LLM-Chain
        docs = docsearch.get_relevant_documents(question)
        answer2 = chain.run(input_documents=docs, question=question)
        print(f'{Fore.YELLOW}Answer with Context-Aware LLM:\n{answer2.strip()}\n==={Style.RESET_ALL}')
        # Context-Aware Prompt-Engineered LLM
        answer3 = "Under-Construction!!!!"
        print(
            f'{Fore.MAGENTA}Answer with Context-Aware Prompt-Engineered LLM:\n{answer3.strip()}\n===\n{Style.RESET_ALL}')

        ###
        print('Do you want to ask another question : Y/N ?\n===')

        to_continue = input()
        if to_continue == 'Y':
            print('Going to another QA round')
        elif to_continue == 'N':
            print('Exiting!')
            break
        else:
            print(f'Unknown value = {to_continue}, going to another QA round')
