from langchain import OpenAI

if __name__ == '__main__':
    print("Loading LLM Model ###")
    model_name = "text-davinci-003"
    temperature = 0.2
    llm = OpenAI(model_name=model_name, temperature=temperature)
    question = "How can I be happy?"
    answer = llm.predict(question)
    print(f'question = {question}')
    print(f"answer = {answer}")
