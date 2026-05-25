"""Provider interface for LLM communication."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from internal.schema import Message, ToolDefinition


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the contract for communicating with LLMs.

    Using Protocol allows for structural subtyping - any class with
    a matching generate method signature will satisfy this interface.
    """

    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition],
    ) -> Message:
        """Send context history and available tools to the LLM for inference.

        Args:
            messages: Current context history.
            available_tools: List of tools the model can call.

        Returns:
            The model's response message.

        Raises:
            ProviderError: If the LLM request fails.
        """
        ...


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Inherit from this class if you prefer explicit inheritance
    over Protocol-based duck typing.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition],
    ) -> Message:
        """Send context history and available tools to the LLM for inference.

        Args:
            messages: Current context history.
            available_tools: List of tools the model can call.

        Returns:
            The model's response message.

        Raises:
            ProviderError: If the LLM request fails.
        """
        pass


class ProviderError(Exception):
    """Raised when an LLM provider request fails."""

    pass