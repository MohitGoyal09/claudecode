# Claudecode

A terminal-based AI coding agent built in Python. Streams LLM responses in real-time, executes tools (file reading, shell commands, editing), and renders everything through a rich terminal UI.

## Architecture

```
main.py          ── CLI entry point, orchestrates Agent + TUI
agent/           ── Core agent loop, event definitions, session management
client/          ── OpenRouter LLM client (streaming + non-streaming)
context/         ── Conversation context & message management
tools/           ── Pluggable tool framework (read_file, shell, edit, ...)
ui/              ── Rich-based terminal UI (panels, syntax highlighting)
utils/           ── Path resolution, text processing, token counting
prompts/         ── System prompt generation
config/          ── Configuration loading and validation
```

### Event flow

```
User input → CLI → Agent.run() → LLM Client.chat_completion()
                        ↓
              yields AgentEvent stream
                        ↓
              CLI._process_message() → TUI renders panels
```

Every LLM response and tool execution is streamed as typed `AgentEvent` dataclass instances. The CLI listens to the event stream and delegates to the TUI for rendering — meaning the UI never lags behind the actual work.

## Quick start

### Prerequisites

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (free tier available)

### Setup

```bash
# Clone
git clone <repo-url>
cd claudecode

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API access
cp .env.example .env
# Edit .env with your OpenRouter key
```

### Usage

```bash
python main.py "Your prompt here"
```

Example:

```bash
python main.py "read main.py and explain what it does"
```

Or run in interactive mode:

```bash
python main.py
```

The agent will respond with streaming text output. For file/tool requests, you'll see a tool-call panel appear, execute, and display results inline.

## Project Structure

```
claudecode/
├── main.py                  # CLI entry point (Click + asyncio)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
│
├── agent/
│   ├── agent.py             # Core agent loop, tool orchestration
│   ├── events.py            # AgentEvent types (TEXT_DELTA, TOOL_CALL, etc.)
│   └── session.py           # Session management (client, tools, context)
│
├── client/
│   ├── llm_client.py        # OpenRouter API client (streaming + retries)
│   └── response.py          # Response types (StreamEvent, TokenUsage, ToolCall)
│
├── context/
│   └── manager.py           # Conversation context & message history
│
├── tools/
│   ├── base.py              # Abstract Tool, ToolInvocation, ToolResult
│   ├── registry.py          # ToolRegistry (register, lookup, invoke)
│   └── builtin/             # Built-in tools
│       ├── read_file.py     # Read file tool
│       ├── write_file.py    # Write file tool
│       ├── edit_file.py     # Edit file tool
│       ├── shell.py         # Shell command execution
│       ├── list_dir.py      # Directory listing
│       ├── grep.py          # Search files by pattern
│       ├── glob.py          # Glob pattern matching
│       ├── todo.py          # Task management
│       ├── memory.py        # Persistent memory storage
│       ├── web_search.py    # Web search
│       └── web_fetch.py    # Web content fetching
│
├── ui/
│   └── tui.py               # Rich terminal UI (panels, syntax highlight)
│
├── utils/
│   ├── paths.py             # Path resolution, binary detection
│   └── text.py              # Text truncation, token counting
│
├── prompts/
│   └── system.py            # System prompt builder
│
└── config/
    ├── config.py            # Configuration models
    └── loader.py            # Configuration loading
```

## Key Design Decisions

**Event-driven architecture.** The agent yields structured events (`AgentEvent`) rather than printing directly. This decouples the core logic from the UI — the same agent could drive a web UI, Slack bot, or CI pipeline without changes.

**Streaming-first.** Every LLM response streams token-by-token through the UI. Tool calls also stream via `TOOL_CALL_START → TOOL_CALL_COMPLETE` events, so the user sees what's happening in real-time.

**Pluggable tools.** Tools inherit from `Tool` (abstract base) and register with `ToolRegistry`. Adding a new tool requires: a params Pydantic model, an `execute()` method, and one line to register it. Tool schemas are auto-generated via Pydantic's `model_json_schema()`.

**Retry with backoff.** The LLM client retries on rate limits, connection errors, and API errors with exponential backoff (up to 3 attempts).

**Session-based.** Each interaction creates a `Session` that manages the LLM client, tool registry, and conversation context. Sessions have unique IDs for tracking.

## Configuration

### Environment Variables

| Environment Variable  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `OPENROUTER_API_KEY`  | OpenRouter API key (required)                                 |
| `OPENROUTER_BASE_URL` | OpenRouter base URL (default: `https://openrouter.ai/api/v1`) |

### Configuration File

The agent uses a `config.yaml` file in the working directory with the following options:

```yaml
model:
  name: "minimax/minimax-m2.5:free"  # Model to use
  temperature: 1.0                     # Sampling temperature (0-2)
  context_window: 256000             # Model context window size

cwd: "."                              # Working directory

shell_environment:
  ignore_default_excludes: false
  exclude_patterns:                   # Patterns to exclude from shell output
    - "*KEY*"
    - "*TOKEN*"
    - "*SECRET*"
  set_vars: {}                        # Environment variables to set

max_turns: 100                        # Maximum agent turns per message
max_tool_output_tokens: 50000        # Max tokens for tool output truncation

developer_instructions: null         # Additional system instructions
user_instructions: null              # Additional user instructions

debug: false                          # Enable debug mode
```

## Dependencies

| Package         | Purpose                                           |
| --------------- | ------------------------------------------------- |
| `openai`        | OpenRouter API (OpenAI-compatible)                |
| `click`         | CLI argument parsing                              |
| `rich`          | Terminal UI (panels, syntax highlighting, themes) |
| `pydantic`      | Data validation & JSON schema generation          |
| `python-dotenv` | Environment variable loading                      |
| `tiktoken`      | Token counting                                    |
| `httpx`         | HTTP client for web tools                         |

## Status

Working features:

- [x] Streaming LLM responses via OpenRouter
- [x] Rich terminal UI with assistant/tool panels
- [x] Built-in tools:
  - [x] `read_file` - Read file contents with syntax highlighting
  - [x] `write_file` - Create/write files
  - [x] `edit_file` - Edit files with diff display
  - [x] `shell` - Execute shell commands
  - [x] `list_dir` - List directory contents
  - [x] `grep` - Search files by pattern
  - [x] `glob` - Glob pattern matching
  - [x] `todo` - Task management
  - [x] `memory` - Persistent memory storage
  - [x] `web_search` - Web search
  - [x] `web_fetch` - Fetch web content
- [x] Tool framework with registry and schema generation
- [x] Conversation context management
- [x] System prompt with security and operational guidelines
- [x] Session management with unique IDs
- [x] Configuration via YAML file
- [x] Token counting and output truncation

## License

MIT