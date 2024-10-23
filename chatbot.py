import sys
import os
import time
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
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from message_retriever import MessageRetriever
from slack_menu_builder import SlackMenuBuilder
from typing import List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from pymongo.results import UpdateResult
from bson.objectid import ObjectId
from http.server import HTTPServer, SimpleHTTPRequestHandler
from aiohttp import web
from slack_sdk.web import SlackResponse
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError 

#TODO: remove
rag_chain: Runnable = Field(default_factory=Runnable)

class SlackChatBot():

    contextualize_system_prompt = (
        """You are an assistant for question-answering tasks.
        Use only the following pieces of retrieved context to answer
        the question and nothing else. If you don't know the answer, say that you
        don't know. Format your answers in pretty Slack Block UI form.
        """)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    title_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a short title name to summarize the question below. Preferably keep it less than 5 words with no punctuation"),
        ("human", "{input}"),
    ])

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
    chats_collection_name = "Chats"
    users_collection_name = "Users"

    slack_bot_user_id = "U07P9UX792P"
    try:
        slack_token = os.environ.get("SLACK_BOT_TOKEN")
        slack_client = AsyncWebClient(token=slack_token)
    except Exception as error:
        print("An exception occurred getting slack client:", error)

    basic_words_file_path = "basic_words.txt"

    port = "80"

    user_id: str = Field(default_factory=str)
    chat_id: ObjectId = Field(default_factory=ObjectId)

    llm = ChatOpenAI(model="gpt-4o-mini")

    message_retriever = MessageRetriever(mongo_client=mongo_client,
                                         basic_words_file_path=basic_words_file_path,
                                         database_name=database_name,
                                         documents_collection_name=documents_collection_name,
                                         search_keys_collection_name=search_keys_collection_name)
    retriever_chain = create_history_aware_retriever(llm, message_retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    global rag_chain
    rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)

    slack_menu_builder = SlackMenuBuilder()

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

    async def switch_chat(self, chat_id: ObjectId) -> None:
        if self.chat_id == chat_id:
            return

        self.chat_id = chat_id
        self.app_config = {"configurable": {"thread_id": str(self.chat_id)}}

        database = self.mongo_client[self.database_name]

        query_filter = { "_id": self.user_id }
        update_operation = { "$set": { "chat_id": self.chat_id } }
        await database[self.users_collection_name].update_one(query_filter, update_operation)

        query_filter = { "_id": self.chat_id }
        find_chat = await database[self.chats_collection_name].find_one(query_filter)
        if not find_chat:
            await database[self.chats_collection_name].insert_one({ "_id": self.chat_id, "user_id": self.user_id, "title": "", "chat_history": [], "ts": time.time() })
        else:
            for message in find_chat["chat_history"]:
                message_to_add: BaseMessage
                match(message["type"]):
                    case "human":
                        message_to_add = HumanMessage(message["content"])

                    case "ai":
                        message_to_add = AIMessage(message["content"])

                self.app.update_state(self.app_config, {"chat_history": message_to_add})

    async def set_chat_history(self, chat_history: List[str]) -> UpdateResult:
        database = self.mongo_client[self.database_name]
        query_filter = { "_id": self.chat_id }
        update_operation = { "$set": { "chat_history": chat_history } }
        return await database[self.chats_collection_name].update_one(query_filter, update_operation)

    async def set_config(self, user_id: str, chat_id: ObjectId = None) -> None:
        if (self.user_id == user_id and not chat_id) or self.chat_id == chat_id:
            return

        if self.user_id != user_id:
            self.user_id = user_id

            database = self.mongo_client[self.database_name]
            query_filter = { "_id": self.user_id }
            user_obj = await database[self.users_collection_name].find_one(query_filter)

            if not user_obj:
                chat_id = ObjectId()
                await database[self.users_collection_name].insert_one({"_id": self.user_id})
            elif not chat_id:
                chat_id = user_obj["chat_id"]

        await self.switch_chat(chat_id=chat_id)

    def serialize_chat_history(self,) -> List[str]:
        return [message.to_json().get("kwargs") for message in self.app.get_state(self.app_config).values["chat_history"]]

    async def send_slack_chat_history(self, user_id:str, channel_id: str) -> SlackResponse:
        await self.set_config(user_id)

        state = self.app.get_state(self.app_config).values
        blocks = []

        if "chat_history" not in state:
            slack_response = await self.slack_client.chat_postMessage(
                channel=channel_id,
                text=f"chat history is empty"
            )
        else:
            messages_fallback = []
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

                messages_fallback.append(message.pretty_repr())

            slack_response = await self.slack_client.chat_postMessage(
                channel=channel_id,
                blocks=blocks,
                text='\n'.join(messages_fallback),
            )

    async def answer_query(self, user_id: str, message: str, channel_id: str) -> None:
        await self.set_config(user_id)

        state = self.app.get_state(self.app_config).values
        chat_history = state["chat_history"] if "chat_history" in state else None
        if not chat_history:
            output_parser = StrOutputParser()
            chain = self.title_generation_prompt | self.llm | output_parser
            chat_title = await chain.ainvoke({"input": message})

            database = self.mongo_client[self.database_name]
            query_filter = { "_id": self.chat_id }
            update_operation = { "$set": { "title": chat_title } }
            await database[self.chats_collection_name].update_one(query_filter, update_operation)

        llm_response = await self.app.ainvoke(
            {"input": message},
            config=self.app_config,
        )

        slack_response = await self.slack_client.chat_postMessage(
            channel=channel_id,
            text=llm_response["answer"]
        )

        await self.set_chat_history(self.serialize_chat_history())

    async def handle_message(self, request) -> Response:
        data = await request.json()
        event = data["event"]
        user_id = event["user"]
        channel_id = event["channel"]
        message = event["text"]

        if user_id != self.slack_bot_user_id:
            asyncio.create_task(self.answer_query(user_id=user_id, message=message, channel_id=channel_id))

        return web.Response(status=200)

    async def send_slack_menu(self, user_id: str, channel_id: str) -> SlackResponse:
        await self.set_config(user_id)

        database = self.mongo_client[self.database_name]
        query_filter = { "_id": self.chat_id }
        chat_obj = await database[self.chats_collection_name].find_one(query_filter)
        chat_title = chat_obj["title"]

        state = self.app.get_state(self.app_config).values
        chat_history = state["chat_history"] if "chat_history" in state else None

        query_filter = { "user_id": self.user_id }
        chats = await database[self.chats_collection_name].find(query_filter).to_list()
        chats = sorted(chats, key=lambda chat: chat["ts"], reverse=True)

        await self.slack_menu_builder.send(channel_id=channel_id,
                                    chat_title=chat_title,
		                            chat_history=chat_history,
		                            chats=chats)

    async def handle_menu(self, request) -> Response:
        data = await request.post()
        user_id = data["user_id"]
        channel_id = data["channel_id"]

        asyncio.create_task(self.send_slack_menu(user_id=user_id, channel_id=channel_id))

        return web.Response(status=200)

    async def reset_current_chat(self, user_id: str) -> UpdateResult:
        await self.set_config(user_id)

        return await self.set_chat_history([])

    async def add_new_chat(self, user_id: str, message: str, channel_id: str) -> None:
        new_chat_id = ObjectId()
        await self.set_config(user_id, new_chat_id)

        await self.answer_query(user_id=user_id, message=message, channel_id=channel_id)

    async def switch_chat_notify(self, user_id: str, chat_id: ObjectId, channel_id: str) -> None:
        await self.set_config(user_id=user_id, chat_id=chat_id)

        database = self.mongo_client[self.database_name]
        query_filter = { "_id": self.chat_id }
        chat_obj = await database[self.chats_collection_name].find_one(query_filter)
        chat_title = chat_obj["title"]

        slack_response = await self.slack_client.chat_postMessage(
            channel=channel_id,
            blocks=[
                {
			        "type": "section",
			        "text": {
				        "type": "mrkdwn",
				        "text": f"Chat set to `{chat_title}`"
			        }
		        },
            ],
            text=f"Chat set to \"{chat_title}\""
        )

    async def handle_interact(self, request) -> Response:
        data = await request.post()
        payload = json.loads(data["payload"])
        user_id = payload["user"]["id"]
        channel_id = payload["channel"]["id"]
        actions = payload["actions"]
        
        for action in actions:
            action_id = action["action_id"]

            match action_id:
                case "show_full_chat_history":
                    asyncio.create_task(self.send_slack_chat_history(user_id=user_id, channel_id=channel_id))

                case "add_new_chat":
                    message = payload["state"]["values"]["new_chat_input"]["add_new_chat"]["value"]
                    if not message:
                        return web.Response(status=400, reason="new chat requires a query")

                    asyncio.create_task(self.add_new_chat(user_id=user_id, message=message, channel_id=channel_id))

                case "switch_chat":
                    chat_id = None
                    match action["type"]:
                        case "button":
                            chat_id = ObjectId(action["value"])
                        case "static_select":
                            selected_option = action["selected_option"]
                            if selected_option:
                                chat_id = ObjectId(selected_option["value"])

                    if chat_id:
                        asyncio.create_task(self.switch_chat_notify(user_id=user_id, chat_id=chat_id, channel_id=channel_id))

        return web.Response(status=200)

    async def main(self,) -> None:
        web_app = web.Application()
        web_app.router.add_post("/", self.handle_message)
        web_app.router.add_post("/menu", self.handle_menu)
        web_app.router.add_post("/interact", self.handle_interact)

        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()

        print(f"Server started at http://localhost:{self.port}/")
        await asyncio.Event().wait()

if __name__ == "__main__":
    slack_chat_bot = SlackChatBot()
    asyncio.run(slack_chat_bot.main())