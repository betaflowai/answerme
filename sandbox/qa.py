from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

if __name__ == '__main__':
    model_name = "text-curie-001"
    temperature = 0.9
    llm = OpenAI(model_name=model_name, temperature=temperature)
    # --
    # repo_id = "google/flan-t5-xxl"
    # repo_id = "stabilityai/stablelm-tuned-alpha-3b" # TODO, model got stuck !
    # repo_id = "databricks/dolly-v2-3b"
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 64})
    #
    with open("../sample_data/sample1/faq1.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    # query = "Do you allow Cashier's Checks or Money Orders?"  # q1 : original question
    # query = "Is it allowed to use Cashier's Checks?" # q1.1 -> success
    query = "What are the supported methods of payments?"  # q1.2 -> failed

    print(query)
    docs = docsearch.get_relevant_documents(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    # query = "What did the president say about Justice Breyer"
    res = chain.run(input_documents=docs, question=query)
    print(res)
