from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError 
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
from fuzzy import Soundex
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from bson.objectid import ObjectId
import re
import os
import fuzzy
import json
import asyncio

slack_bot_user_id = "U07P9UX792P"
slack_bot_token_key = "SLACK_BOT_TOKEN"
num_messages_request = 200
basic_words_file_path = "basic_words.txt"
doc_store_path = "doc_store.json"
search_dict_path = "search_dict.json"
slack_channel_types = "public_channel"
database_name = "SlackbotData"
documents_collection_name = "Documents"
search_keys_collection_name = "SearchKeys"

class MessageRetriever(BaseRetriever):
    num_relevant_docs: int = 5
    proximity: int = 1
    searched_doc_value: int = 1
    proximity_doc_value: int = 0.1
    basic_words: set[str] = Field(default_factory=set)
    soundex:Soundex = fuzzy.Soundex(4)
    mongo_client: AsyncIOMotorClient = Field(default_factory=AsyncIOMotorClient)


    async def add_channel_messages_to_database(self, slack_client: WebClient, channel_id: str, word_keys: set[str]) -> None:
        cursor = None

        while True:
            try:
                channel_history = slack_client.conversations_history(channel=channel_id, limit=num_messages_request, cursor=cursor)
            except SlackApiError as e:
                print(f"Error fetching messages: {e.response['error']}")
                return

            messages = channel_history["messages"]

            for message in messages:
                database = self.mongo_client[database_name]

                doc_id = ObjectId()
                messageText = message["text"]
                metadata = {"channel_id": channel_id, "user": message["user"], "ts": message["ts"]}
                await database[documents_collection_name].insert_one({"_id": doc_id, "page_content": messageText, "metadata": metadata})

                local_word_keys = set()
                for word in re.split(r"[ .;,?!\"]", messageText):
                    word = word.lower().encode('ascii', 'ignore')

                    if not word or word in self.basic_words:
                        continue

                    word_key = self.soundex(word)

                    if not word_key in word_keys:
                        word_keys.add(word_key)
                        await database[search_keys_collection_name].insert_one({"_id": word_key, "doc_ids": [doc_id]})
                            
                    local_word_keys.add(word_key)

                query_filter = { "_id": { "$in": list(local_word_keys)} }
                update_operation = { "$push": { "doc_ids": doc_id } }
                await database[search_keys_collection_name].update_many(query_filter, update_operation)

            response_metadata = channel_history["response_metadata"]

            if not response_metadata:
                return

            cursor = response_metadata["next_cursor"]

    async def empty_collection(self, database: AsyncIOMotorDatabase, collection_name: str) -> None: 
        await database[collection_name].drop()
        await database.create_collection(collection_name)

    async def update_database(self) -> None:
        slack_token = os.environ.get(slack_bot_token_key)

        if not slack_token:
            print(f"{slack_bot_token_key} envar was not found")
            return

        try:
            slack_client = WebClient(token=slack_token)
        except Exception as error:
            print("An exception occurred getting slack client:", error)

        cursor = None

        try:
            result = slack_client.users_conversations(user=slack_bot_user_id, cursor=cursor, types=slack_channel_types)
        except SlackApiError as e:
            print(f"Error fetching channels list: {e.response['error']}")
            return

        database = self.mongo_client[database_name]
        await self.empty_collection(database=database, collection_name=documents_collection_name)
        await self.empty_collection(database=database, collection_name=search_keys_collection_name)

        channels = result["channels"]

        word_keys = set()
        for channel in channels:
            await self.add_channel_messages_to_database(slack_client=slack_client, channel_id=channel["id"], word_keys=word_keys)

    def __init__(self, mongo_client: AsyncIOMotorClient) -> None:
        super().__init__()

        self.mongo_client = mongo_client

        try:
            with open(basic_words_file_path, 'r') as file:
                self.basic_words = {word.strip() for word in file}
        except FileNotFoundError:
            print(f"Error: {basic_words_file_path} not found.")

    def get_relevant_documents(self, query: str) -> List[Document]:
        raise Exception("Message Retriever is meant to be used aync. Use ainvoke instead")

    def handle_query_doc_occurrence(self, query_doc_occurrences: defaultdict[str, float], doc_ids: List[int], doc_collection: AsyncIOMotorCollection):
        for doc_id in doc_ids:
            # TODO: add proximity
            query_doc_occurrences[doc_id] += self.searched_doc_value


    async def aget_relevant_documents(self, query: str,) -> List[Document]:
        query_doc_occurrences = defaultdict(float)

        database = self.mongo_client[database_name]
        doc_collection = database[documents_collection_name]

        for word in re.split(r"[ .;,?!\"]", query):
            if not word or word in self.basic_words:
                continue

            word_key = self.soundex(word)
            search_key_obj = await database[search_keys_collection_name].find_one({ "_id": word_key })

            if search_key_obj:
                self.handle_query_doc_occurrence(query_doc_occurrences=query_doc_occurrences, doc_ids=search_key_obj["doc_ids"], doc_collection=doc_collection)

        max_size = self.num_relevant_docs * (self.proximity * 2 + 1)
        relevant_doc_ids = sorted(query_doc_occurrences, key=query_doc_occurrences.get, reverse=True)[:max_size]

        relevant_doc_objs = await doc_collection.find( { "_id": { "$in": list(relevant_doc_ids)} } ).to_list()
        relevant_docs = [Document(id=doc_obj["_id"], page_content=doc_obj["page_content"], metadata=doc_obj["metadata"]) for doc_obj in relevant_doc_objs]
        return relevant_docs