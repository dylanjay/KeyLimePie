import os
from typing import List
from langchain_core.messages import BaseMessage
from pydantic import Field
from slack_sdk.web import SlackResponse
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

class SlackMenuBuilder():
	try:
		slack_token = os.environ.get("SLACK_BOT_TOKEN")
		slack_client = AsyncWebClient(token=slack_token)
	except Exception as error:
		print("An exception occurred getting slack client:", error)

	blocks: List[dict] = []

	def build_history_option(self, message: BaseMessage) -> dict:
		if message.type == "human":
			return {
				"type": "rich_text_quote",
				"elements": [
					{
						"type": "text",
						"text": message.content
					}
				]
			}
		elif message.type == "ai":
			return {
				"type": "rich_text_section",
				"elements": [
					{
						"type": "text",
						"text": message.content
					}
				]
			}

	def build_history(self, chat_history: List[BaseMessage]) -> None:
		elements = []
		if chat_history:
			for message in chat_history:
				elements.append(self.build_history_option(message))
		else:
			elements = [
				{
					"type": "rich_text_section",
					"elements": [
						{
							"type": "emoji",
							"name": "ghost"
						},
						{
							"type": "text",
							"text": " Chat history empty. Nothing to see here! ",
							"style": {
								"bold": True
							}
						},
						{
							"type": "emoji",
							"name": "ghost"
						}
					]
				}
			]

		self.blocks.append(
			{
				"type": "rich_text",
				"elements": elements
			}
		)

		if chat_history:
			self.blocks.append(
				{
					"type": "actions",
					"elements": [
						{
							"type": "button",
							"text": {
								"type": "plain_text",
								"text": "Show full chat history",
								"emoji": True
							},
							"value": "None",
							"action_id": "show_full_chat_history"
						}
					]
				}
			)

	def build_chat_option(self, chat_title: str, chat_id: str) -> None:
		self.blocks.append(
			{
				"type": "section",
				"text": {
					"type": "mrkdwn",
					"text": chat_title
				},
				"accessory": {
					"type": "button",
					"action_id": "switch_chat",
					"text": {
						"type": "plain_text",
						"emoji": True,
						"text": "Switch"
					},
					"value": chat_id
				}
			}
		)

	def build_chat_select_option(self, chat_title: str, chat_id: str) -> dict:
		return {
			"text": {
				"type": "plain_text",
				"text": chat_title,
				"emoji": True
			},
			"value": chat_id
		}

	def build_chat_options(self, chats: List[dict]) -> None:
		if chats:
			options = []
			for chat in chats:
				chat_title = chat["title"]
				if not chat_title:
					continue

				chat_id = str(chat["_id"])

				if (len(options) < 3):
					self.build_chat_option(chat_title=chat_title, chat_id=chat_id)

				options.append(self.build_chat_select_option(chat_title=chat_title, chat_id=chat_id))

			if options:
				self.blocks.append(
					{
						"type": "input",
						"element": {
							"type": "static_select",
							"placeholder": {
								"type": "plain_text",
								"text": "Select a chat to switch to",
								"emoji": True
							},
							"options": options,
							"action_id": "switch_chat"
						},
						"label": {
							"type": "plain_text",
							"text": "More Chats",
							"emoji": True
						}
					}
				)
			

	def build_header(self, text: str) -> None:
		self.blocks.append(
			{
				"type": "header",
				"text": {
					"type": "plain_text",
					"text": text
				}
			}
		)

	def build_new_line(self,) -> None:
		self.blocks.append(
			{
				"type": "section",
				"text": {
					"type": "plain_text",
					"text": "\n"
				}
			}
		)

	def build_divider(self,) -> None:
		self.blocks.append(
			{
				"type": "divider"
			}
		)

	def build_menu(self, chat_title: str, chat_history: List[BaseMessage], chats: List[dict]) -> None:
		self.blocks =[]

		self.build_header("Chat Menu")
		self.blocks.append(
			{
				"type": "section",
				"text": {
					"type": "mrkdwn",
					"text": ":arrow_down: :arrow_down: :arrow_down: :arrow_down: :arrow_down:"
				}
			}
		)
		self.build_divider()
		self.build_header(chat_title if chat_title else "No Current Chat")
		self.build_history(chat_history)
		self.build_new_line()
		self.build_divider()
		self.build_new_line()

		self.blocks.append(
			{
				"type": "input",
				"block_id": "new_chat_input",
				"dispatch_action": False,
				"element": {
					"type": "plain_text_input",
					"multiline": True,
					"action_id": "add_new_chat",
				},
				"label": {
					"type": "plain_text",
					"text": "New Chat"
				},
			}
		)

		self.blocks.append(
			{
				"type": "actions",
				"elements": [
					{
						"type": "button",
						"action_id": "add_new_chat",
						"text": {
							"type": "plain_text",
							"text": "Add new chat",
							"emoji": True
						},
						"value": "new_chat_input",
					}
				]
			}
		)

		self.build_new_line()
		self.build_divider()
		self.build_header("Previous Chats")
		self.build_chat_options(chats)

	async def send(self,
		  channel_id: str,
		  chat_title: str,
		  chat_history: List[BaseMessage],
		  chats: List[dict],) -> SlackResponse:

		self.build_menu(chat_title=chat_title, chat_history=chat_history, chats=chats)
		slack_response = await self.slack_client.chat_postMessage(
            channel=channel_id,
            blocks=self.blocks,
			text="Unable to build slack menu"
        )

		return slack_response

