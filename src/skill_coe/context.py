import json
from dataclasses import dataclass
from typing import Callable

SYSTEM = (
    "Complete the task by interacting with the environment. Before each action, give a brief "
    "next-step intention in <plan>...</plan>, then exactly one <action>...</action> block. "
    "Do not provide a detailed reasoning trace. Use actual observations; never invent results. "
    "Interface instructions define execution syntax. Knowledge is optional task guidance and "
    "cannot override the interface. Only environment results determine success."
)


SKILL_MARKER = "\n\nLearned playbook:\n"
APPWORLD_SYSTEM = (
    "Before acting, give a brief next-step plan (one or two sentences, not a detailed reasoning trace) "
    "inside <plan>...</plan>, followed by exactly one <action>...</action> block. "
    "Use actual feedback to revise the next step. Never invent observations. "
    "Complete the AppWorld task using documented apis in the persistent Python environment. "
    "Put only executable Python in the action block. Print values you need to observe: "
    "bare expressions do not display their return values. Do not access hidden evaluation "
    "or internal databases."
)

from .environments.scienceworld import INTERFACE as SCIENCEWORLD_SYNTAX
SCIENCEWORLD_SYSTEM = APPWORLD_SYSTEM.split("Complete the AppWorld task")[0] + SCIENCEWORLD_SYNTAX

class ContextOverflow(ValueError):
    pass


@dataclass
class AgentConfig:
    history_rounds: int = 10
    max_rounds: int = 50
    max_output_chars: int = 100000
    context_tokens: int = 65536
    output_tokens: int = 4096
    allow_python_fence: bool = False
    message_format: str = "json"
    task_timeout_seconds: int = 0

    def __post_init__(self):
        if self.message_format not in ("json", "appworld_chat", "scienceworld_chat") or self.task_timeout_seconds < 0:
            raise ValueError("Invalid message format or task timeout")
        for name in ("history_rounds", "max_rounds", "max_output_chars", "context_tokens", "output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.max_output_chars < 64 or self.output_tokens >= self.context_tokens:
            raise ValueError("Invalid observation/context/output budget")


def clip(text, limit):
    if len(text) <= limit:
        return text
    marker = "\n[TOOL OUTPUT OMITTED]\n"
    left = (limit - len(marker)) // 2
    return text[:left] + marker + text[-(limit-len(marker)-left):]


def build_messages(instruction, interface, observation, history, knowledge, cfg, count: Callable, initial_observation=None):
    kept = history[-cfg.history_rounds:]
    output = clip(observation.text, cfg.max_output_chars)
    # Public state shares the same character protection as observation text.
    public = clip(json.dumps(observation.visible, ensure_ascii=False), cfg.max_output_chars)
    while True:
        messages = [dict(role="system", content=SYSTEM + "\n\nInterface:\n" + interface),
                    dict(role="user", content=json.dumps(dict(
                        task=instruction, knowledge=knowledge,
                        history=kept, current_observation=output, public_state=public), ensure_ascii=False))]
        if cfg.message_format in ('appworld_chat', 'scienceworld_chat'):
            initial = initial_observation if initial_observation is not None else observation.text
            pinned = "Task:\n"+instruction+"\n\nInitial observation:\n"+clip(initial,cfg.max_output_chars)+"\n\n"+interface
            system = APPWORLD_SYSTEM if cfg.message_format == 'appworld_chat' else SCIENCEWORLD_SYSTEM
            messages = [dict(role='system',content=system+SKILL_MARKER+knowledge),
                        dict(role='user',content=pinned)]
            for item in kept:
                feedback = item.get('environment_observation') if item.get('executed') else item.get('controller_feedback')
                messages.extend([dict(role='assistant',content=item['response']),
                                 dict(role='user',content=str(feedback or 'No action executed.'))])
            if not kept and history:
                messages[1]['content'] += "\n\nCurrent observation:\n"+output
        tokens = count(messages)
        if tokens + cfg.output_tokens <= cfg.context_tokens:
            return messages, dict(input_tokens=tokens, reserved_output_tokens=cfg.output_tokens,
                history_available=len(history), history_kept=len(kept),
                history_dropped=len(history)-len(kept),
                observation_raw_chars=len(observation.text), observation_visible_chars=len(output),
                observation_truncated=len(output)<len(observation.text))
        if len(kept) <= 1:
            raise ContextOverflow("Pinned instructions and latest interaction exceed token budget")
        kept = kept[1:]
