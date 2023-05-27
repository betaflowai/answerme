# https://python.langchain.com/en/latest/modules/models/getting_started.html
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.llms import AlephAlpha
from langchain.llms import AI21

if __name__ == '__main__':
    # openai models
    # https://platform.openai.com/docs/models/overview
    model_name = "text-curie-001"
    temperature = 0.9
    #
    # https://python.langchain.com/en/latest/modules/models/llms/getting_started.html
    print('Exercise # 1 : testing llm prediction : diff models and temperatures ')

    # llm = OpenAI(model_name=model_name, temperature=temperature)
    #
    # TODO Investigate why HuggingFace hub models gets stuck ( timeout, never got a response)
    #   https://github.com/hwchase17/langchain/issues/3275
    #   managed to make it work with a hack (changed model to xxl version, temp=0.7
    #       and chunk_size limited)
    #   https://github.com/hwchase17/langchain/issues/3275#issuecomment-1544790326
    #   But, need to find the root cause
    #   https://github.com/hwchase17/langchain/issues/3275#issuecomment-1537432279
    #
    # test HuggingFace different models
    # repo_id = "google/flan-t5-xxl"
    # repo_id = "stabilityai/stablelm-tuned-alpha-3b" # TODO, model got stuck !
    # repo_id = "databricks/dolly-v2-3b"
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 64})
    # llm = AI21()

    llm = AlephAlpha(model="luminous-extended", maximum_tokens=20, stop_sequences=["Q:"])
    print(f'llm-model = \n {llm}')
    n_repeat = 5
    question = 'How can I be Happy ?'
    print(f'making {n_repeat} calls for llm.predict to answer the question : {question}')
    for i in range(n_repeat):
        answer = llm.predict(question)
        print(f'Answer at call # {i + 1} to llm.predict = {answer.strip()}')
    print('====')
    # Note how results vary with high temperature
    print('Exercise 2 : test llm.generate')
    llm_result = llm.generate(["Tell me a joke"] * 15)
    print(f'len of llm.generate results = {len(llm_result.generations)}')
    # Note how results vary with high temperature
    print(f'generations[0] = \n{llm_result.generations[0]}')
    print(f'generations[-1] = \n{llm_result.generations[-1]}')
    print('Get provider specific information')
    print(llm_result.llm_output)
    #
    print('Exercise # 3 : Get the predicted number of tokens a priori , it costs money!')
    text = "This is the most silly joke I have ever heard of!"
    n = llm.get_num_tokens(text)
    print(f'Num of tokens for text : {text} = {n}')
    #
    print('Exercises finished!')
    #
