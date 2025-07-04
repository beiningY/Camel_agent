from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    r"""An abstract base class for all agents."""

    @abstractmethod
    def load_env(self) -> Any:
        r"""Loads the environment variables."""
        pass

    @abstractmethod
    def load_config(self) -> Any:
        r"""Loads the configuration."""
        pass

    @abstractmethod
    def init_agent(self) -> Any:
        r"""Initializes the agent."""
        pass
    