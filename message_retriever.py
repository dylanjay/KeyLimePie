import re
import os
import fuzzy
import asyncio
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from collections import defaultdict
from fuzzy import Soundex
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

class MessageRetriever(BaseRetriever):
    
    num_relevant_docs: int = 5
    proximity: int = 1
    searched_doc_value: int = 1
    proximity_doc_value: int = 0.1

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

    def get_relevant_documents(self, query: str) -> List[Document]:
        raise Exception("Message Retriever is meant to be used aync. Use ainvoke instead")

    def handle_query_doc_occurrence(self, query_doc_occurrences: defaultdict[str, float], doc_ids: List[int], doc_collection: AsyncIOMotorCollection):
        for doc_id in doc_ids:
            # TODO: add proximity
            query_doc_occurrences[doc_id] += self.searched_doc_value

    async def aget_relevant_documents(self, query: str,) -> List[Document]:
        query_doc_occurrences = defaultdict(float)

        database = self.mongo_client[self.database_name]
        doc_collection = database[self.documents_collection_name]

        ascii_text = query.encode('ascii', 'ignore').decode('ascii')
        re.sub(r'[^a-zA-Z0-9 \n]+', '', ascii_text)

        for word in re.split(r"[ \n]", ascii_text):
            word = word.lower()

            if not word or word in self.basic_words:
                continue

            word_key = self.soundex(word)

            search_key_obj = await database[self.search_keys_collection_name].find_one({ "_id": word_key })

            if search_key_obj:
                self.handle_query_doc_occurrence(query_doc_occurrences=query_doc_occurrences, doc_ids=search_key_obj["doc_ids"], doc_collection=doc_collection)

        relevant_doc_objs = await doc_collection.find( { "_id": { "$in": list(query_doc_occurrences.keys())} } ).to_list()

        #max_size = self.num_relevant_docs * (self.proximity * 2 + 1)
        relevant_doc_objs = sorted(relevant_doc_objs, key=lambda doc_obj: query_doc_occurrences[doc_obj["_id"]], reverse=True)

        relevancy_total = sum(query_doc_occurrences.values())

        relevant_docs = []
        for doc_obj in relevant_doc_objs:
            id = doc_obj["_id"]
            page_content = doc_obj["page_content"]
            metadata = doc_obj["metadata"]
            relevancy_count = query_doc_occurrences[id]
            metadata["relevancy_count"] = relevancy_count
            metadata["relevancy_index"] = relevancy_count / relevancy_total
            relevant_docs.append(Document(id=id, page_content=page_content, metadata=metadata))

        return relevant_docs