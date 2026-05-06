from agent.agent import Agent
import asyncio
import click
from typing import Any
import sys
from agent.events import AgentEventType
from ui.tui import TUI, get_console


console = get_console()
class CLI:

    def __init__(self):
        self.agent : Agent | None = None
        self.tui = TUI(console)


    async def run_single(self , message : str):
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)
    
    async def _process_message(self , message : str) -> str | None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content" ,"")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)
            
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error")
                console.print(f"[error]Error: {error}[/error]")
        return final_response
        



async def run(messages: list[dict[str, Any]]):
   pass


@click.command()
@click.argument("prompt" , required = False)
def main(prompt: str | None = None):
    cli = CLI()
    # messages = [
    #     {"role" : "system" , "content" : "You are a helpful assistant."},
    #     {"role" : "user" , "content" : prompt}
    # ]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))

        if result is None:
            sys.exit(1)


   
    


main()