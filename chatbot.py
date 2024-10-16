import sys
import os
import aioconsole
import asyncio
from collections import defaultdict
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

    slack_bot_user_id = "U07P9UX792P"
    slack_bot_token_key = "SLACK_BOT_TOKEN"
    basic_words_file_path = "basic_words.txt"

    llm = ChatOpenAI(model="gpt-4o-mini")

    #TODO: remove this
    user_id = "test"
    #thread_id: str = Field(default_factory=str)
    #app_config: dict[dict[str, str]] = Field(default_factory=dict[dict[str, str]])
    thread_id = "test"
    app_config = {"configurable": {"thread_id": thread_id}}

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
    app = state_graph.compile(checkpointer=MemorySaver())

    async def activate_thread(self, thread_id: str) -> None:
        if not thread_id:
            user_input = await aioconsole.ainput("Input thread name: ")
            print()
            self.thread_id = user_input.lower()
        else:
            self.thread_id = thread_id
        self.app_config = {"configurable": {"thread_id": self.thread_id}}

        database = self.mongo_client[self.database_name]
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

    async def chatbot(self,):

        await self.activate_thread(thread_id="test")

        print(self.welcome_prompt)

        while True:
            user_input = await aioconsole.ainput("query: ")
            user_input = user_input.lower()
            print()

            match(user_input):
                case "/bye":
                    print("Goodbye!")
                    break

                case "/thread":
                    await self.activate_thread()

            response = await self.app.ainvoke(
                {"input": user_input},
                config=self.app_config,
            )

            database = self.mongo_client[self.database_name]
            query_filter = { "user_id": self.user_id, "thread_id": self.thread_id }
            serializable_chat_history = [message.to_json().get("kwargs") for message in response["chat_history"]]
            update_operation = { "$set": { "chat_history": serializable_chat_history } }
            database[self.threads_collection_name].update_one(query_filter, update_operation)

            print(response["answer"])
            print()

    async def main(self,) -> None:
        await self.chatbot()

if __name__ == "__main__":
    slack_chat_bot = SlackChatBot()
    asyncio.run(slack_chat_bot.main())