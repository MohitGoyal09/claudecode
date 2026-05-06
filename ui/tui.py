from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text


AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)

_console: Console | None = None

def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME , highlight=False)
    return _console

class TUI:
    def __init__(self , console: Console | None = None) -> None:
        self.console = console or get_console()
        self._assistant_stream_open = False
   
    def begin_assistant(self) -> None:
       self.console.print()
       self.console.print(Rule(Text("Assistant" , style = "assistant")))
       self._assistant_stream_open = True

    def end_assistant(self) -> None:
        if self._assistant_stream_open:
          self.console.print()
        self._assistant_stream_open = False

    def stream_assistant_delta(self , content : str) -> None:
        self.console.print(content , end = "" , markup = False)
