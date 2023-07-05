import argparse
import sys
import time
from datetime import datetime
import logging
import ntpath

import requests


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-path', type=str, required=True, help='pdf file')
    parser.add_argument('--question', type=str, required=True, help='question')
    parser.add_argument('--x-rapidapi-key', type=str, required=True, help='rapid-api app key')
    parser.add_argument('--x-rapidapi-host', type=str, required=True, help='rapid-api host')
    parser.add_argument('--rapidapi-base-url', type=str, required=True, help='rapidapi base url')
    parser.add_argument('--sleep-time', type=int, required=True, help='time to sleep')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    args = get_parser().parse_args()
    logger.info(f'Args = {args}')

    headers = {
        "X-RapidAPI-Key": args.x_rapidapi_key,
        "X-RapidAPI-Host": args.x_rapidapi_host
    }

    # 1) Get pre-signed url
    pre_signed_url_endpoint = args.rapidapi_base_url + "/psurl"
    file_basename = ntpath.basename(args.input_file_path)
    file_basename_without_ext = file_basename.split(".")[0]
    file_basename_ext = file_basename.split(".")[1]

    file_bin = open(args.input_file_path, 'rb').read()
    time_stamp = datetime.now().isoformat()
    file_new_basename = f"{file_basename_without_ext}_{time_stamp}.{file_basename_ext}"
    pre_signed_url_response = requests.get(url=pre_signed_url_endpoint, headers=headers,
                                           params={'filename': file_new_basename})
    logger.info(f'Sent pre-signed-request : {pre_signed_url_response.url}')
    logger.info(f'Return status code = {pre_signed_url_response.status_code}')
    if pre_signed_url_response.ok:
        logger.info(f'Pre-signed-url request was successful')
    else:
        logger.error(f'Return text = {pre_signed_url_response.text},exiting!')
        sys.exit(-1)

    # 2) Upload file
    upload_url = pre_signed_url_response.json()["url"]
    logger.info(f'Uploading file with url = {upload_url}')
    upload_response = requests.put(url=upload_url, data=file_bin)
    logger.info(f'Upload response status-code = {upload_response.status_code}')
    if upload_response.ok:
        logger.info(f'Upload was run successfully')
    else:
        logger.error(f'Upload had error {upload_response.text}, exiting')
        sys.exit(-1)
    # 3) Wait for indexing
    logger.info(f'Sleeping for {args.sleep_time} to wait for the document to be indexed')
    time.sleep(args.sleep_time)
    # 4) Ask questions
    params = {'question': args.question, 'filename': file_new_basename}
    qa_endpoint = args.rapidapi_base_url + "/answerme"
    qa_response = requests.get(url=qa_endpoint, headers=headers, params=params)
    if qa_response.ok:
        logger.info(f'QA request was successful')
        logger.info(f'QA response = \n{qa_response.json()}')
    else:
        logger.info(f'QA request failed with status code = {qa_response.status_code} and text {qa_response.text}')
