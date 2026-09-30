from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Observation:
    text: str
    instruction: str = ""
    interface: str = ""
    terminated: bool = False
    truncated: bool = False
    success: Optional[bool] = None
    reward: Optional[float] = None
    score: Optional[float] = None
    environment_calls: int = 0
    api_calls: int = 0
    # Visible fields are prompt context; diagnostics are for logging only.
    visible: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    finish_reason: str = "stop"
    usage: Dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    benchmark: str
    def reset(self, task: Dict[str, Any], seed: int) -> Observation: ...
    def step(self, action: str) -> Observation: ...
    def evaluate(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...


class Model(Protocol):
    def count_tokens(self, messages: List[Dict[str, str]]) -> int: ...
    def generate(self, messages: List[Dict[str, str]], *, max_tokens: int, seed: int) -> Generation: ...
