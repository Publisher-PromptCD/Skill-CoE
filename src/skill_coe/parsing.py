"""Extract explicit actions only; never execute inferred natural-language commands."""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedAction:
    action: Optional[str]
    status: str


def parse_action(text: str, allow_python_fence: bool = False) -> ParsedAction:
    starts, ends = text.count("<action>"), text.count("</action>")
    if starts > 1 or ends > 1:
        return ParsedAction(None, "multiple_actions")
    if starts or ends:
        match = re.search(r"<action>(.*?)</action>", text, re.S)
        if starts != 1 or ends != 1 or not match:
            return ParsedAction(None, "malformed_action")
        action = match.group(1).strip()
        if not action:
            return ParsedAction(None, "empty_action")
        return ParsedAction(action, "action_block")
    if allow_python_fence:
        fences = re.findall(r"```([^\n`]*)\n(.*?)```", text, re.S)
        if len(fences) > 1:
            return ParsedAction(None, "multiple_code_blocks")
        if len(fences) == 1 and fences[0][0].strip().lower() in ("python", "py"):
            if text.count("```") != 2:
                return ParsedAction(None, "malformed_code_block")
            code = fences[0][1].strip()
            return ParsedAction(code or None, "python_fence" if code else "empty_action")
    return ParsedAction(None, "missing_action")
