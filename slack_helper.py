import os
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
from slack_sdk.web import SlackResponse
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
from yarl import cache_clear 

try:
    slack_user_token = os.environ.get("SLACK_USER_TOKEN")
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_client = AsyncWebClient(token=slack_user_token)
except Exception as error:
    print("An exception occurred getting slack client:", error)

llm = ChatOpenAI(model="gpt-4o-mini")

#channel_id = "C07LN74LP1V"
channel_id = "D07PYRNKY1W"

async def count_messages() -> int:
    cursor = None
    num_messages = 0

    while True:
        try:
            channel_history = await slack_client.conversations_history(channel=channel_id, cursor=cursor)
        except SlackApiError as e:
            print(f"Error fetching messages: {e.response['error']}")
            return

        num_messages += len(channel_history["messages"])

        response_metadata = channel_history["response_metadata"]

        if not response_metadata:
            print(num_messages)
            return num_messages

        cursor = response_metadata["next_cursor"]


async def clear_channel() -> None:
    cursor = None

    while True:
        try:
            channel_history = await slack_client.conversations_history(channel=channel_id, cursor=cursor)
        except SlackApiError as e:
            print(f"Error fetching messages: {e.response['error']}")
            return

        messages = channel_history["messages"]

        for message in messages:
            result = await slack_client.chat_delete(
                token=slack_user_token,
                channel=channel_id,
                ts=message["ts"],
            )

        response_metadata = channel_history["response_metadata"]

        if not response_metadata:
            return

        cursor = response_metadata["next_cursor"]


async def simulate_messages() -> None:
    output = await llm.ainvoke([HumanMessage(content="Write me 100 messages you might see in a social slack channel. keep each message around 3 sentences long. no message indices. with first and last names at the start of the message delimited by :. some of the messages should relate or reply to each other")])

    for line in output.content.split("\n\n"):
        slack_response = await slack_client.chat_postMessage(
        channel=channel_id,
        text=line,
        )

async def list_channels() -> None:
    result = await slack_client.conversations_list()
    for channel in result["channels"]:
        print(channel["name"])



if __name__ == "__main__":
    asyncio.run(clear_channel())
    #asyncio.run(simulate_messages())
    #asyncio.run(count_messages())
    #asyncio.run(list_channels())