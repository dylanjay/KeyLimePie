import sys
import os
import aioconsole
import asyncio
import http
import socketserver
import json
from collections import defaultdict
from httpcore import Response
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from message_retriever import MessageRetriever
from typing import List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from http.server import HTTPServer, SimpleHTTPRequestHandler
from aiohttp import web
from slack_sdk import WebClient 
from slack_sdk.errors import SlackApiError 

#TODO: remove
rag_chain: Runnable = Field(default_factory=Runnable)

class SlackChatBot():

    contextualize_system_prompt = (
        """You are an assistant for question-answering tasks.
        Use only the following pieces of retrieved context to answer
        the question and nothing else. If you don't know the answer, say that you
        don't know.
        """)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    welcome_prompt = (
"""Welcome to the chatbot!
Type '/bye' to quit
Type '/thread' for a new thread
Type '/reset' to reset the current thread
"""
)

    try:
        db_pw_key = "SLACK_RAG_CLUSTER_PW"
        db_pw = os.environ.get(db_pw_key)
        db_URI_key = "SLACK_RAG_CLUSTER_URI"
        db_URI = os.environ.get(db_URI_key).format(db_pw=db_pw)
        mongo_client = AsyncIOMotorClient(db_URI, server_api=ServerApi('1'))
    except Exception as error:
        print("An exception occurred setting up mongo client:", error)

    database_name = "SlackbotData"
    documents_collection_name = "Documents"
    search_keys_collection_name = "SearchKeys"
    threads_collection_name = "Threads"
    users_collection_name = "Users"

    slack_bot_user_id = "U07P9UX792P"
    try:
        slack_token = os.environ.get("SLACK_BOT_TOKEN")
        slack_client = WebClient(token=slack_token)
    except Exception as error:
        print("An exception occurred getting slack client:", error)

    basic_words_file_path = "basic_words.txt"

    port = "80"

    user_id: str = Field(default_factory=str)
    thread_id: str = Field(default_factory=str)

    llm = ChatOpenAI(model="gpt-4o-mini")

    message_retriever = MessageRetriever(mongo_client=mongo_client,
                                         basic_words_file_path=basic_words_file_path,
                                         database_name=database_name,
                                         documents_collection_name=documents_collection_name,
                                         search_keys_collection_name=search_keys_collection_name)
    retriever_chain = create_history_aware_retriever(llm, message_retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    global rag_chain
    rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)

    class State(TypedDict):
        input: str
        chat_history: Annotated[Sequence[BaseMessage], add_messages]
        context: str
        answer: str

    async def call_model(state: State):
        response = await rag_chain.ainvoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
    
    state_graph = StateGraph(state_schema=State)
    state_graph.add_edge(START, "model")
    state_graph.add_node("model", call_model)
    app_config: str = Field(default_factory=str)
    app = state_graph.compile(checkpointer=MemorySaver())

    async def activate_thread(self, thread_id: str) -> None:
        if self.thread_id is thread_id:
            return

        self.thread_id = thread_id
        self.app_config = {"configurable": {"thread_id": self.thread_id}}

        database = self.mongo_client[self.database_name]

        query_filter = { "_id": self.user_id }
        update_operation = { "$set": { "thread_id": self.thread_id } }
        await database[self.users_collection_name].update_one(query_filter, update_operation)

        query_filter = { "user_id": self.user_id, "thread_id": self.thread_id }
        find_thread = await database[self.threads_collection_name].find_one(query_filter)
        if not find_thread:
            await database[self.threads_collection_name].insert_one({"user_id": self.user_id, "thread_id": self.thread_id, "chat_history": []})
        else:
            for message in find_thread["chat_history"]:
                message_to_add: BaseMessage
                match(message["type"]):
                    case "human":
                        message_to_add = HumanMessage(message["content"])

                    case "ai":
                        message_to_add = AIMessage(message["content"])

                self.app.update_state(self.app_config, {"chat_history": message_to_add})

    async def set_chat_history(self, chat_history: List[str]) -> None:
        database = self.mongo_client[self.database_name]
        query_filter = { "user_id": self.user_id, "thread_id": self.thread_id }
        update_operation = { "$set": { "chat_history": chat_history } }
        await database[self.threads_collection_name].update_one(query_filter, update_operation)

    async def detect_config_change(self, user_id: str) -> None:
        if not self.user_id or not self.thread_id or user_id != self.user_id:
            self.user_id = user_id

            database = self.mongo_client[self.database_name]
            filter = { "_id": self.user_id }
            user_obj = await database[self.users_collection_name].find_one(filter)

            if user_obj:
                thread_id = user_obj["thread_id"]
            else:
                thread_id = str(ObjectId())
                await database[self.users_collection_name].insert_one({"_id": self.user_id})

            await self.activate_thread(thread_id=thread_id)

    def serialize_chat_history(self,) -> List[str]:
        return [message.to_json().get("kwargs") for message in self.app.get_state(self.app_config).values["chat_history"]]

    async def answer_query(self, data) -> None:
        event = data["event"]
        channel_id = event["channel"]
        message = event["text"]
        user_id = event["user"]

        await self.detect_config_change(user_id)

        llm_response = await self.app.ainvoke(
            {"input": message},
            config=self.app_config,
        )

        slack_response = self.slack_client.chat_postMessage(
            channel=channel_id,
            text=llm_response["answer"]
        )

        await self.set_chat_history(self.serialize_chat_history())

    async def handle_message(self, request) -> Response:
        data = await request.json()
        event = data["event"]
        user_id = event["user"]

        if user_id != self.slack_bot_user_id:
            asyncio.create_task(self.answer_query(data))

        return web.Response(status=200)

    async def handle_thread(self, request) -> Response:
        data = await request.json()

        await self.detect_config_change(request)

        return web.Response(status=200)

    async def handle_reset(self, request) -> Response:
        data = await request.json()
        event = data["event"]
        user_id = event["user"]

        await self.detect_config_change(user_id)

        await self.set_chat_history([])

        return web.Response(status=200)

    async def send_slack_chat_history(self, data) -> None:
        channel_id = data["channel_id"]
        user_id = data["user_id"]

        await self.detect_config_change(user_id)

        state = self.app.get_state(self.app_config).values
        blocks = []
        for message in state["chat_history"]:
            blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"==== {message.type.title()} Message ====="
                }
            })

            blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": message.content
                }
            })

        slack_response = self.slack_client.chat_postMessage(
            channel=channel_id,
            blocks=blocks,
            text=message.pretty_repr(),
        )

    async def handle_print(self, request) -> Response:
        data = dict(await request.post())
        asyncio.create_task(self.send_slack_chat_history(data))

        return web.Response(status=200)


    async def main(self,) -> None:
        web_app = web.Application()
        web_app.router.add_post("/", self.handle_message)
        web_app.router.add_post("/thread", self.handle_thread)
        web_app.router.add_post("/reset", self.handle_reset)
        web_app.router.add_post("/print", self.handle_print)

        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()

        print(f"Server started at http://localhost:{self.port}/")
        await asyncio.Event().wait()

if __name__ == "__main__":
    slack_chat_bot = SlackChatBot()
    asyncio.run(slack_chat_bot.main())