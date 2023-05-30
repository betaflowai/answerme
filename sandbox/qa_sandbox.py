from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

if __name__ == '__main__':
    model_name = "text-curie-001"
    temperature = 0.7
    llm = OpenAI(model_name=model_name, temperature=temperature)
    # --
    # repo_id = "google/flan-t5-xxl"
    # repo_id = "stabilityai/stablelm-tuned-alpha-3b" # TODO, model got stuck !
    # repo_id = "databricks/dolly-v2-3b"
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 64})
    #
    # context_file = open("../sample_data/sample1/faq1.txt", "r")
    # context_text = context_file.read()
    context_text = \
        """
        Q:What is your return and refund policy?
        \n\n
        A : You can return you order with full refund in up to 15 days after the delivery day.
        If you want to return it after 15 days and up to 100 days since delivery you eligible for a partial refund.
        After that the package is not returnable.
        \n\n
        """
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_text(context_text)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings,
                                  metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    # query = "Do you allow Cashier's Checks or Money Orders?"  # q1 : original question
    # query = "Is it allowed to use Cashier's Checks?" # q1.1 -> success
    query1 = "How can I return my order"  # q1.2 -> failed
    query2 = "Can I return my order after 100 days?"
    query = query2
    docs = docsearch.get_relevant_documents(query)
    #
    res_without_context = llm.predict(query)
    #
    chain = load_qa_chain(llm, chain_type="stuff")
    res_with_context_without_sources = chain.run(input_documents=docs, question=query)
    #
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    res_with_context_with_sources = chain({"input_documents": docs, "question": query}, return_only_outputs=False)
    #
    print(f"query = {query}")
    print(f'Answer without context = \n{res_without_context}')
    print(f'Answer with context without sources = \n{res_with_context_without_sources}')
    print(f"Answer with context with sources = \n{res_with_context_with_sources['output_text']}")
