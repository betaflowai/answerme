import os
import sys

import requests

if __name__ == '__main__':
    url = "https://mcw0rn675i.execute-api.eu-north-1.amazonaws.com/DEV/answerme"

    # question = "How was the passenger car demand in 2021?"
    while True:
        print('input question')
        question = input()
        if question == "exit":
            print('Exiting')
            sys.exit(0)
        print('input filename')
        filename = input()  # "HY_2021_e.pdf"
        api_key = os.getenv('QA_API_KEY_1')
        r = requests.get(url=url, headers={'x-api-key': api_key}, params={"question": question, "filename": filename})
        if r.ok:
            print('request-response operation ended successfully')
            print(r.json())
        else:
            print(r.text)
        pass
