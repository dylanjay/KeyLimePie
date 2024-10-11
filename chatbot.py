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
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from message_retriever import MessageRetriever
from typing import List, Optional
from pydantic import Field
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi

db_URI = "mongodb+srv://dylan:MSNYJMzVmbB8Z08E@slackragcluster.0lzyr.mongodb.net/?retryWrites=true&w=majority&appName=SlackRagCluster"
mongo_client = AsyncIOMotorClient(db_URI, server_api=ServerApi('1'))

llm = ChatOpenAI()
output_parser = StrOutputParser()
session_store: dict[str, BaseChatMessageHistory] = Field(default_factory=dict)
message_retriever = MessageRetriever(mongo_client=mongo_client)

system_prompt = (
    """You are an assistant for question-answering tasks.
    Use only the following pieces of retrieved context to answer
    the question and nothing else. If you don't know the answer, say that you
    don't know. Use three sentences maximum and keep the 
    answer concise.
    \n\n
    {context}""")

prompt = ChatPromptTemplate.from_messages(
[
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if not session_id in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

async def chatbot():

    print("Welcome to the chatbot! Type 'bye' to quit.")

    user_input = await aioconsole.ainput("Input name of previous session to continue or a new one:")
    session_id = user_input.lower()

    while True:
        user_input = await aioconsole.ainput("")
        user_input = user_input.lower()

        if user_input == "bye":
            print("Goodbye!")
            break

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(message_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input",
        output_messages_key="answer",
        )

        response = await conversational_rag_chain.ainvoke(
            {"input": user_input},
            config=
            {
                "configurable": {"session_id": session_id}
            },
        )["answer"]

        print(response)

async def main() -> None:
    await message_retriever.update_database(mongo_client=mongo_client)

    #await chatbot()

if __name__ == "__main__":
    asyncio.run(main())