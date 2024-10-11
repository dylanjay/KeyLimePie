""" Basic operations using Slack_sdk """

import os
from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError 
import PyPDF2
import sys
import asyncio

channel_id = "C07PNH08QPP"

waitSeconds = 1.1

def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):

            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            text.append(page_text)
    return text

def remove_all_character(text, character):
    alteredText = text.replace(character, "")

    if (not alteredText or alteredText.isspace()):
        return ""

    return alteredText

def delete_all_messages(client):
    result = client.conversations_history(channel=channel_id)

    conversation_history = result["messages"]

    for message in conversation_history:
        result = client.chat_delete(
            channel=channel_id,
            ts=message["ts"])
        print(result)

async def wait_and_post_message(client, message, seconds):
    await asyncio.sleep(seconds)

    response = client.chat_postMessage(
        channel="rag", 
        text=message)

    yield response

async def post_all_messages(client, messages):
    for message in messages:
        async for result in wait_and_post_message(client, message, waitSeconds):
            print(result)

if __name__ == "__main__":
    pdf_path = "HarryPotterSorcerersStone.pdf"  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)

    slack_token = os.environ.get('SLACK_BOT_TOKEN')

    client = WebClient(token=slack_token)

    try:
        result = client.conversations_list()

        print(result)


        # allParagraphs = []

        # paragraphIndex = 0

        # startParagraphIndex = 192

        # for text in extracted_text:
        #     paragraphs = text.split('\n')
            
        #     for paragraph in paragraphs:
     
        #         if paragraph:
        #             if paragraphIndex > startParagraphIndex:
        #                 allParagraphs.append(paragraph)

        #         paragraphIndex += 1

        # print(len(allParagraphs))

        # asyncio.run(post_all_messages(client, allParagraphs))

    except SlackApiError as e:
	    assert e.response["error"]