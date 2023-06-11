"""
https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
https://docs.langchain.com/docs/components/chains/index_related_chains
"""
import nltk
import argparse
import logging
import os
import zipfile

import numpy as np
from tqdm import tqdm
import PyPDF2
import magic
from termcolor import colored
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

#
DOUBLE_NEW_LINE_SEP = "\n\n"


def get_text_from_zip_archive(file_path):
    pass


def get_text_from_text_file(file_path):
    logger.info(f'Getting text from {file_path}')
    file_obj = open(file_path)
    logger.info(f'Appending text from {file_path} to context string')
    text = file_obj.read()
    sentences = nltk.sent_tokenize(text)
    sentences_lengths = list(map(lambda s: len(s), sentences))
    logger.debug(
        f'Split text into {len(sentences)} sentences with max-len = {max(sentences_lengths)} '
        f'and average-len = {np.round(np.nanmean(sentences_lengths), 0)}')
    return text


def get_text_from_pdf(file_path):
    logger = logging.getLogger()
    logger.info(f'Converting {file_path} to text, page by page')
    pdf_file_obj = open(file_path, 'rb')
    pdfreader = PyPDF2.PdfReader(pdf_file_obj)
    n = len(pdfreader.pages)
    all_text_list = []
    for i in tqdm(range(n), desc='pdf to text progress in pages'):
        logger.debug(f'\nConverting page {i + 1} to text')
        page = pdfreader.pages[i]
        page_text = page.extract_text()
        sentences = nltk.sent_tokenize(page_text)
        sentences_lengths = list(map(lambda s: len(s), sentences))
        logger.debug(f'Split page {i + 1} into {len(sentences)} sentences, '
                     f'with max-len = {max(sentences_lengths)}')
        all_text_list.append(f"## Page {i + 1} ##{DOUBLE_NEW_LINE_SEP}==={DOUBLE_NEW_LINE_SEP}")
        all_text_list.extend(sentences)

    logger.info(f'Successfully converted all pages in {file_path} to text')
    all_text = DOUBLE_NEW_LINE_SEP.join(all_text_list)
    return all_text


def get_text_from_docx(file_path):
    pass


def get_context_text(file_path):
    logger = logging.getLogger()
    # https://pypi.org/project/python-magic/
    # MIME Type mapping
    # https://learn.microsoft.com/en-us/previous-versions/office/office-2007-resource-kit/ee309278%28v=office.12%29
    file_mime_type = magic.from_buffer(open(file_path, 'rb').read(2048), mime=True)
    context_text = ""
    if file_mime_type == 'application/pdf':
        context_text = get_text_from_pdf(file_path)
    elif file_mime_type == 'text/plain':
        context_text = get_text_from_text_file(file_path)
    elif file_mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        pass
    return context_text
    #
    # context_text = ""
    # with zipfile.ZipFile(file_path, "r") as zip_file:
    #     for file_name in zip_file.namelist():
    #         if file_name.endswith('.txt'):
    #             logger.info(f'Appending text from {os.path.join(file_path, file_name)} to context string')
    #             text = zip_file.read(file_name).decode()
    #             context_text += text
    #             context_text += "\n---\n"
    #         elif file_name.endswith('.pdf'):
    #             logger.info(f'Appending text from {os.path.join(file_path, file_name)} to context string')
    #             # TODO
    #             #   https://stackoverflow.com/questions/62055857/reading-a-pdf-from-a-zipfile
    #         else:
    #             logger.warning(f'file {file_name} is not a text file, and I am not processing it')
    # return context_text


def get_debug_level(level_str):
    if level_str == 'info':
        return logging.INFO
    elif level_str == 'debug':
        return logging.DEBUG
    elif level_str == 'error':
        return logging.ERROR
    elif level_str == 'warn':
        return logging.WARNING
    else:
        raise ValueError(f'Logging level {level_str} not supported')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-file', type=str, required=True, help='Full path for context file')
    parser.add_argument('--log-level', type=str, choices=['info', 'debug', 'warn', 'error'], help='logging level',
                        default='info')
    return parser


if __name__ == '__main__':
    colorama_init()
    nltk.download('punkt')
    #
    parser = get_parser()
    args = parser.parse_args()
    #
    logging.basicConfig(level=get_debug_level(args.log_level))
    logger = logging.getLogger()
    #
    chunk_size = 300
    chunk_overlap = 50

    # build chain
    logger.info("Loading LLM Model ###")
    model_name = "text-davinci-003"
    temperature = 0.2
    llm = OpenAI(model_name=model_name, temperature=temperature)
    context_text = get_context_text(file_path=args.context_file)
    # https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/character_text_splitter.html
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(context_text)
    chain = load_qa_chain(llm, chain_type="stuff")
    embeddings = OpenAIEmbeddings()
    logger.info(f'{Fore.BLUE} Indexing context docs. {Style.RESET_ALL}')
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    logger.info(f'{Fore.CYAN} I am reading to questions, HIT ME !')
    while True:
        # print(colored('Input question', 'red'))

        logger.info(f"{Fore.GREEN}Input question : {Style.RESET_ALL}!")
        question = input()
        # Generic LLM
        answer1 = llm.predict(question)
        logger.info(f'{Fore.BLUE}Answer with Generic LLM:\n{answer1.strip()}\n==={Style.RESET_ALL}')
        # Context-Aware LLM-Chain
        docs = docsearch.get_relevant_documents(question)
        answer2 = chain.run(input_documents=docs, question=question)
        logger.info(f'{Fore.YELLOW}Answer with Context-Aware LLM:\n{answer2.strip()}\n==={Style.RESET_ALL}')
        # Context-Aware Prompt-Engineered LLM
        answer3 = "Under-Construction!!!!"
        logger.info(
            f'{Fore.MAGENTA}Answer with Context-Aware Prompt-Engineered LLM:\n{answer3.strip()}\n===\n{Style.RESET_ALL}')

        ###
        logger.info('Do you want to ask another question : Y/N ?\n===')

        to_continue = input()
        if to_continue == 'Y':
            print('Going to another QA round')
        elif to_continue == 'N':
            print('Exiting!')
            break
        else:
            print(f'Unknown value = {to_continue}, going to another QA round')
