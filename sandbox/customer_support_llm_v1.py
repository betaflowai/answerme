"""
https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
"""
import logging

from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def get_context_text():
    files = ["../sample_data/vfuk/esim_page.txt", "../sample_data/vfuk/users.txt"]
    all_text = ""
    for file in files:
        f = open(file)
        text = f.read()
        all_text += f"\n---\n {text}"
    return all_text


if __name__:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    print("Loading LLM Model")
    # build chain
    model_name = "text-davinci-003"
    temperature = 0.2
    llm = OpenAI(model_name=model_name, temperature=temperature)
    context_text = get_context_text()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_text(context_text)
    chain = load_qa_chain(llm, chain_type="stuff")
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    while True:
        print('input question')
        question = input()
        answer_with_no_context = llm.predict(question)
        print(f'Answer 1:\n{answer_with_no_context.strip()}')
        docs = docsearch.get_relevant_documents(question)
        answer_with_context = chain.run(input_documents=docs, question=question)
        print(f'Answer 2:\n{answer_with_context.strip()}')
        print('Do you want to ask another question : Y/N ?')
        to_continue = input()
        if to_continue == 'Y':
            print('Going to another QA round')
        elif to_continue == 'N':
            print('Exiting!')
            break
        else:
            print(f'Unknown value = {to_continue}, going to another QA round')
