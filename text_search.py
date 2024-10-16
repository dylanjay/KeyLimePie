import re
import os
import fuzzy
import asyncio
from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError 
from typing import List, Any, Optional
from collections import defaultdict
from fuzzy import Soundex
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from bson.objectid import ObjectId

class TextSearch():
    num_messages_request = 200
    slack_channel_types = "public_channel"

    basic_words: set[str] = Field(default_factory=set)
    mongo_client: AsyncIOMotorClient = Field(default_factory=AsyncIOMotorClient)
    database_name: str = Field(default_factory=str)
    documents_collection_name: str = Field(default_factory=str)
    search_keys_collection_name: str = Field(default_factory=str)

    soundex:Soundex = fuzzy.Soundex(4)

    def __init__(self, mongo_client: AsyncIOMotorClient, basic_words_file_path, database_name, documents_collection_name, search_keys_collection_name,) -> None:
        super().__init__()

        self.mongo_client = mongo_client
        self.database_name = database_name
        self.documents_collection_name = documents_collection_name
        self.search_keys_collection_name = search_keys_collection_name

        try:
            with open(basic_words_file_path, 'r') as file:
                self.basic_words = {word.strip() for word in file}
        except FileNotFoundError:
            print(f"Error: {basic_words_file_path} not found.")

    async def add_channel_messages_to_database(self, slack_client: WebClient, channel_id: str, word_keys: set[str]) -> None:
        cursor = None

        while True:
            try:
                channel_history = slack_client.conversations_history(channel=channel_id, limit=self.num_messages_request, cursor=cursor)
            except SlackApiError as e:
                print(f"Error fetching messages: {e.response['error']}")
                return

            messages = channel_history["messages"]

            for message in messages:
                database = self.mongo_client[self.database_name]

                doc_id = ObjectId()
                messageText = message["text"]
                metadata = {"channel_id": channel_id, "user": message["user"], "ts": message["ts"]}
                await database[self.documents_collection_name].insert_one({"_id": doc_id, "page_content": messageText, "metadata": metadata})

                local_word_keys = set()
                for word in re.split(r"[ .;,?!\"]", messageText):
                    word = word.lower().encode('ascii', 'ignore')

                    if not word or word in self.basic_words:
                        continue

                    word_key = self.soundex(word)

                    if not word_key in word_keys:
                        word_keys.add(word_key)
                        await database[self.search_keys_collection_name].insert_one({"_id": word_key, "doc_ids": [doc_id]})
                            
                    local_word_keys.add(word_key)

                query_filter = { "_id": { "$in": list(local_word_keys)} }
                update_operation = { "$push": { "doc_ids": doc_id } }
                await database[self.search_keys_collection_name].update_many(query_filter, update_operation)

            response_metadata = channel_history["response_metadata"]

            if not response_metadata:
                return

            cursor = response_metadata["next_cursor"]

    async def empty_collection(self, database: AsyncIOMotorDatabase, collection_name: str) -> None: 
        await database[collection_name].drop()
        await database.create_collection(collection_name)

    async def update_database(self) -> None:
        try:
            slack_token = os.environ.get(self.slack_bot_token_key)
            slack_client = WebClient(token=slack_token)
        except Exception as error:
            print("An exception occurred getting slack client:", error)

        cursor = None

        try:
            result = slack_client.users_conversations(user=self.slack_bot_user_id, cursor=cursor, types=self.slack_channel_types)
        except SlackApiError as e:
            print(f"Error fetching channels list: {e.response['error']}")
            return

        database = self.mongo_client[self.database_name]
        await self.empty_collection(database=database, collection_name=self.documents_collection_name)
        await self.empty_collection(database=database, collection_name=self.search_keys_collection_name)

        channels = result["channels"]

        word_keys = set()
        for channel in channels:
            await self.add_channel_messages_to_database(slack_client=slack_client, channel_id=channel["id"], word_keys=word_keys)