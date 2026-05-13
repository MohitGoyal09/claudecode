from datetime import datetime
import uuid
from client.llm_client import LLMClient
from config.config import Config
from context.manager import ContextManager
from tools.registry import create_default_registery


class Session:
    def __init__(self, config: Config):
        self.config = config
        self.client = LLMClient(config=config)
        self.tool_registry = create_default_registery()
        self.context_manager: ContextManager | None = ContextManager(config=config)

        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        self._turn_count = 0

    def increment_turn(self) -> int:
        self._turn_count += 1
        self.updated_at = datetime.now()

        return self._turn_count
