from pathlib import Path
from typing import Any
from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.console import Group
from rich.syntax import Syntax
from rich.markdown import Markdown
from config.config import Config
from tools.base import ToolConfirmation
from utils.paths import display_path_rel_to_cwd
import re

from utils.text import truncate_text

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
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class TUI:
    def __init__(self, config: Config, console: Console | None = None) -> None:
        self.console = console or get_console()
        self.config = config
        self.cwd = self.config.cwd
        self._assistant_stream_open = False
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        self._max_block_tokens: int = 25000
        self._tool_start_preview_tokens: int = 4000

    def begin_assistant(self) -> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True

    def end_assistant(self) -> None:
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple]:
        _PREFERRED_ORDER = {
            "read_file": ["path", "offset", "limit"],
            "write_file": ["path", "create_directories", "content"],
            "edit": ["path", "replace_all", "old_string", "new_string"],
            "shell": ["command", "timeout", "cwd"],
            "list_dir": ["path", "include_hidden"],
            "grep": ["path", "case_insensitive", "pattern"],
            "glob": ["path", "pattern"],
            "todos": ["id", "action", "content"],
            "memory": ["action", "key", "value"],
        }

        preferred = _PREFERRED_ORDER.get(tool_name, [])
        ordered: list[tuple[str, Any]] = []
        seen = set()

        for key in preferred:
            if key in args:
                ordered.append((key, args[key]))
                seen.add(key)

        remaining_keys = set(args.keys() - seen)
        ordered.extend((key, args[key]) for key in remaining_keys)

        return ordered

    def _render_args_table(self, tool_name: str, args: dict[str, Any]) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted", justify="right", no_wrap=True)
        table.add_column(style="code", overflow="fold")

        path_hint = args.get("path") if isinstance(args.get("path"), str) else None

        for key, value in self._ordered_args(tool_name, args):
            if isinstance(value, str) and key in {"content", "old_string", "new_string"}:
                lang = (
                    self._guess_language(path_hint)
                    if key == "content"
                    else "diff"
                )
                preview = truncate_text(
                    value,
                    self.config.model_name,
                    self._tool_start_preview_tokens,
                )
                table.add_row(
                    key,
                    Syntax(
                        preview,
                        lang,
                        theme="monokai",
                        word_wrap=True,
                    ),
                )
                continue

            if isinstance(value, bool):
                value = str(value)

            table.add_row(key, value)

        return table

    def stream_assistant_delta(self, content: str) -> None:
        self.console.print(content, end="", markup=False)

    def tool_call_start(
        self, call_id: str, name: str, tool_kind: str | None, arguments: dict[str, Any]
    ) -> None:
        self._tool_args_by_call_id[call_id] = arguments
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"

        title = Text.assemble(
            ("⏺ ", "muted"),
            (name, "tool"),
            ("  ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )
        display_args = dict(arguments)
        for key in ("path", "cwd"):
            val = display_args.get(key)
            if isinstance(val, str) and self.cwd:
                display_args[key] = str(display_path_rel_to_cwd(val, self.cwd))

        panel = Panel(
            (
                self._render_args_table(name, display_args)
                if display_args
                else Text(
                    "(no args)",
                    style="muted",
                )
            ),
            title=title,
            title_align="left",
            subtitle=Text("running", style="muted"),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    def _extract_read_file_code(self, text: str) -> tuple[int, str] | None:
        body = text
        header_match = re.match(r"^Showing lines (\d+)-(\d+) of (\d+)\n\n", text)
        if header_match:
            body = text[header_match.end() :]

        code_lines: list[str] = []
        start_line: int | None = None

        for line in body.splitlines():
            m = re.match(r"^\s*(\d+)\|(.*)$", line)
            if not m:
                return None
            line_no = int(m.group(1))
            m.group(2)
            if start_line is None:
                start_line = line_no
            code_lines.append(m.group(2))

        if start_line is None:
            return None

        return start_line, "\n".join(code_lines)

    def _guess_language(self, path: str | None) -> str:
        if not path:
            return "text"
        suffix = Path(path).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "jsx",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".json": "json",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".css": "css",
            ".html": "html",
            ".xml": "xml",
            ".sql": "sql",
        }.get(suffix, "text")

    def print_welcome(self, title: str, lines: list[str]) -> None:
        body = "\n".join(lines)
        self.console.print(
            Panel(
                Text(body, style="code"),
                title=Text(title, style="highlight"),
                title_align="left",
                border_style="border",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def tool_call_complete(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        success: bool,
        output: str,
        error: str | None,
        metadata: dict[str, Any] | None,
        exit_code : int | None,
        truncated: bool = False,
        diff: str | None = None,
       
    ) -> None:

        border_style = f"tool.{tool_kind}" if tool_kind else "tool"
        status_icon = "✓" if success else "✗"
        status_style = "success" if success else "error"

        title = Text.assemble(
            (f"{status_icon} ", status_style),
            (name, "tool"),
            ("  ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )
        

        args = self._tool_args_by_call_id.get(call_id , {})

        primary_path = None
        blocks = []
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            primary_path = metadata.get("path")

            if name == "read_file" and success:
                if primary_path:
                    extracted = self._extract_read_file_code(output)
                    if extracted:
                        start_line, code = extracted
                        shown_start = metadata.get("shown_start")
                        shown_end = metadata.get("shown_end")
                        total_lines = metadata.get("total_lines")
                        pl = self._guess_language(primary_path)

                        header_parts = [display_path_rel_to_cwd(primary_path, self.cwd)]
                        header_parts.append(" • ")

                        if shown_start and shown_end and total_lines:
                            header_parts.append(
                                f"lines {shown_start}-{shown_end} of {total_lines}"
                            )

                        header = "".join(header_parts)
                        blocks.append(Text(header, style="muted"))
                        blocks.append(
                            Syntax(
                                code,
                                pl,
                                theme="monokai",
                                line_numbers=True,
                                start_line=start_line,
                                word_wrap=False,
                            )
                        )
                    else:
                        output_display = truncate_text(
                            output,
                            self.config.model_name,
                            self._max_block_tokens,
                        )
                        blocks.append(
                            Syntax(
                                output_display,
                                "text",
                                theme="monokai",
                                word_wrap=False,
                            )
                        )
                else:
                    output_display = truncate_text(
                        output,
                        self.config.model_name,
                        self._max_block_tokens,
                    )
                    blocks.append(
                        Syntax(
                            output_display,
                            "text",
                            theme="monokai",
                            word_wrap=False,
                        )
                    )
            elif name in {"write_file", "edit"} and success:
                output_line = output.strip() if output.strip() else "Completed"
                blocks.append(Text(output_line, style="muted"))
                if diff and diff.strip():
                    diff_display = truncate_text(
                        diff,
                        self.config.model_name,
                        self._max_block_tokens,
                    )
                    blocks.append(
                        Syntax(
                            diff_display,
                            "diff",
                            theme="monokai",
                            word_wrap=True,
                        )
                    )
                else:
                    blocks.append(
                        Text(
                            "(no diff text returned; see message above)",
                            style="muted",
                        )
                    )
            elif name == 'shell':
                command = args.get('command')
                if isinstance(command , str) and command.strip():
                    blocks.append(Text(f'$ {command.strip()}', style = 'muted'))
                
                if exit_code is not None:
                    blocks.append(Text(f'exit_code={exit_code}' , style = 'muted'))
                
                output_display = truncate_text(output , self.config.model_name , self._max_block_tokens)
                blocks.append(
                        Syntax(
                            output_display,
                            "text",
                            theme="monokai",
                            word_wrap=True,
                        )
                    )
        else:
            output_display = truncate_text(
                output, self.config.model_name, self._max_block_tokens
            )
            blocks.append(
                Syntax(output_display, "text", theme="monokai", word_wrap=False)
            )

        panel = Panel(
            Group(*blocks),
            title=title,
            title_align="left",
            subtitle=Text("done" if success else "failed", style=status_style),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
