from client.llm_client import LLMClient
import asyncio
import click
from typing import Any


class CLI:

    def __init__(self):
        pass

    def run_single(self):
        pass 


async def run(messages: list[dict[str, Any]]):
    client = LLMClient()
    async for event in client.chat_completion(messages , stream = True):
        print(event)

@click.command()
@click.argument("prompt" , required = False)
def main(prompt: str | None = None):
    messages = [
        {"role" : "system" , "content" : "You are a helpful assistant."},
        {"role" : "user" , "content" : prompt}
    ]
    asyncio.run(run(messages))
    print("done")


main()