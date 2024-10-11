from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError 
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
from fuzzy import Soundex
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from bson.objectid import ObjectId
import re
import os
import fuzzy
import json
import uuid
import asyncio

slack_bot_user_id = "U07P9UX792P"
slack_bot_token_key = "SLACK_BOT_TOKEN"
num_messages_request = 200
basic_words_file_path = "basic_words.txt"
doc_store_path = "doc_store.json"
search_dict_path = "search_dict.json"
slack_channel_types = "public_channel"

class MessageRetriever(BaseRetriever):
    num_relevant_docs: int = 5
    proximity: int = 1
    searched_doc_value: int = 10
    proximity_doc_value: int = 1
    basic_words: set[str] = Field(default_factory=set)
    soundex:Soundex = fuzzy.Soundex(4)
    mongo_client: AsyncIOMotorClient


    def add_channel_messages(self, slack_client: WebClient, docs: dict[str, Document], search_dict: dict[str, set[str]], channel_id: str) -> None:
        cursor = None

        while True:
            try:
                result = slack_client.conversations_history(channel=channel_id, limit=num_messages_request, cursor=cursor)
            except SlackApiError as e:
                print(f"Error fetching messages: {e.response['error']}")
                return

            messages = result["messages"]

            for message in messages:
                messageText = message["text"]
                metadata = {"channel_id": channel_id, "user": message["user"], "ts": message["ts"]}

                doc_id = str(uuid.uuid4())
                docs[doc_id] = Document(id=doc_id, page_content=messageText, metadata=metadata)

                for word in re.split(r"[ .;,?!\"]", messageText):
                    word = word.lower().encode('ascii', 'ignore')

                    if not word or word in self.basic_words:
                        continue

                    key = self.soundex(word)

                    if key not in search_dict:
                        search_dict[key] = set()
                    search_dict[key].add(doc_id)      

            response_metadata = result["response_metadata"]

            if not response_metadata:
                return

            cursor = response_metadata["next_cursor"]

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

        channels = result["channels"]

        for channel in channels:
            self.add_channel_messages(slack_client, channel["id"])

        
        database = self.mongo_client["SlackbotData"]
        try:
            await database["Documents"].drop()
            collection = await database.create_collection("Documents")

            docs = dict()
            serializable_docs = [{"_id": doc_id, "page_content": doc.page_content, "metadata": doc.metadata} for doc_id, doc in docs.items()]
            await collection.insert_many(serializable_docs)
        except Exception as error:
            print("An exception occurred updating docs db:", error)

        try:
            await database["SearchKeys"].drop()
            collection = await database.create_collection("SearchKeys")

            search_dict = dict()
            search_keys = [{"_id": key, "doc_ids": list(value)} for key, value in search_dict.items()]
            await collection.insert_many(search_keys)
        except Exception as error:
            print("An exception occurred updating search keys db:", error)

    def __init__(self, mongo_client: AsyncIOMotorClient) -> None:
        super.__init__()

        self.mongo_client = mongo_client

        try:
            with open(basic_words_file_path, 'r') as file:
                self.basic_words = {word.strip() for word in file}
        except FileNotFoundError:
            print(f"Error: {basic_words_file_path} not found.")


    async def initialize(self, mongo_client: AsyncIOMotorClient) -> None:

        database = mongo_client["SlackbotData"]

        try:
            db_docs = await database["Documents"].find().to_list()
            self.docs = {doc["_id"]: Document(id=doc["_id"], page_content=doc["page_content"], metadata=doc["metadata"]) for doc in db_docs}
            print(self.docs)
        except Exception as error:
            print("An exception occurred initializing docs:", error)
        
        try:
            db_search_keys = await database["SearchKeys"].find().to_list()
            self.search_dict = {key["key"]: set(key["documents"]) for key in db_search_keys}
            print(self.search_dict)
        except Exception as error:
            print("An exception occurred initializing search dict:", error)

    def get_relevant_documents(self, query: str) -> List[Document]:
        raise Exception("Message Retriever is meant to be used aync. Use ainvoke instead")

    # async def handle_query_doc_occurrence(self, query_doc_occurrences: defaultdict[str, float], doc_ids: List[int], doc_collection: AsyncIOMotorCollection):
    #     for doc_id in doc_ids:
    #         doc_collection.aggregate([
    #             {
    #                 $setWindowFields: 
    #                 {
    #                     partitionBy: "$channel_id",
    #                     sortBy: { metadata.ts: 1 }
    #                 }
    #             }
    #         ])


    async def aget_relevant_documents(self, query: str,) -> List[Document]:
        query_doc_occurrences = defaultdict(float)

        database = self.mongo_client["SlackbotData"]
        doc_collection = database["Documents"]

        for word in re.split(r"[ .;,?!\"]", query):
            if not word or word in self.basic_words:
                continue

            key = self.soundex(word)

            result = await database["SearchKeys"].find_one(ObjectId(key))

            if result.count() == 0:
                continue

            doc_ids = result["doc_ids"]
            #self.handle_query_doc_occurrence(query_doc_occurrences=query_doc_occurrences, doc_ids=doc_ids, doc_collection=doc_collection)

        relevant_doc_ids = sorted(query_doc_occurrences.items(), key=lambda x: (-x[1], x[0]))

        max_size = self.num_relevant_docs * (self.proximity * 2 + 1)
        relevant_docs = []
        for doc_id in relevant_doc_ids:
            result = await doc_collection.find_one(ObjectId(doc_id))
            relevant_docs.append(Document(id=result["_id"], page_content=result["page_content"], metadata=result["metadata"]))
            if len(relevant_docs) >= max_size:
                break

        return relevant_docs