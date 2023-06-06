"""
https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
https://docs.langchain.com/docs/components/chains/index_related_chains
"""
import argparse
import logging
import os

from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def get_context_text(root_dir):
    context_text = ""
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                print(f'Getting file {file_name} in context')
                file = open(os.path.join(root, file_name))
                text = file.read()
                context_text += text
                context_text += "\n---\n"
    return context_text


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-root', type=str, required=True, help='Root for context text files')
    return parser


if __name__:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    chunk_size = 300
    chunk_overlap = 20
    #
    parser = get_parser()
    args = parser.parse_args()
    # build chain
    print("Loading LLM Model")
    model_name = "text-davinci-003"
    temperature = 0.2
    llm = OpenAI(model_name=model_name, temperature=temperature)
    context_text = get_context_text(root_dir=args.context_root)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(context_text)
    chain = load_qa_chain(llm, chain_type="stuff")
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    # TODO later
    # prompt_examples = examples = [
    #     {
    #         "question": "How can I convert to e-Sim ?",
    #         "answer":
    #             """
    #             Are follow up questions needed here: Yes.
    #             Follow up: What is your e-mail?
    #             Intermediate answer: My e-mail is mbaddar2@gmail.com.
    #             Follow up: What is your phone Model?
    #             Intermediate answer: By looking up in the context, my Phone Model is Apple 12.
    #             Follow up: Is Phone Model Apple 12 compatible with e-Sim?
    #             Intermediate answer: By looking up in the context, Yes Apple 12 is compatible with e-Sim.
    #             So the final answer is: As you phone is Compatible with e-Sim, you can convert to e-Sim easily by
    #             downloading the app.
    #             """
    #     }]
    while True:
        print('input question')
        question = input()
        # Generic LLM
        answer1 = llm.predict(question)
        print(f'Answer with Generic LLM:\n{answer1.strip()}\n===')
        # Context-Aware LLM-Chain
        docs = docsearch.get_relevant_documents(question)
        answer2 = chain.run(input_documents=docs, question=question)
        print(f'Answer with Context-Aware LLM:\n{answer2.strip()}')
        # Context-Aware Prompt-Engineered LLM
        answer3 = "Under-Construction!!!!"
        print(f'Answer with Context-Aware Prompt-Engineered LLM:\n{answer3.strip()}\n===')
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
