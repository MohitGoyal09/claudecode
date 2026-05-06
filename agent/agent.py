from __future__ import annotations
from typing import AsyncGenerator, Type
from client.response import StreamEventType
from .events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from types import TracebackType

class Agent:
    def __init__(self):
        self.client: LLMClient = LLMClient()
    
    async def run(self , message: str):
        yield AgentEvent.agents_start(message)
        final_response: str | None = None

        async for event in self._agentic_loop(message):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agents_end(response = final_response)
    
    async def _agentic_loop(self, user_message: str) -> AsyncGenerator[AgentEvent , None]:
        messages = [
            {"role" : "system" , "content" : "You are a helpful assistant."},
            {"role" : "user" , "content" : user_message}
        ]

        response_text = ""

        async for event in self.client.chat_completion(messages , stream = True):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta is not None:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
                else:
                    yield AgentEvent.text_delta("")
            elif event.type == StreamEventType.MESSAGE_COMPLETE:
                yield AgentEvent.text_complete(event.text_delta.content if event.text_delta else "")
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agents_error(event.error or "Unknown error", details = event.__dict__)

    
    async def __aenter__(self) -> Agent:
        return self
    
    async def __aexit__(self , exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        await self.client.close()