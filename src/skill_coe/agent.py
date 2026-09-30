"""ReAct execution and episode logging."""
import json
from .clock import elapsed_clock
from dataclasses import asdict
from pathlib import Path
from .context import AgentConfig, ContextOverflow, build_messages, clip
from .parsing import parse_action


class Recorder:
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.events = self.path / "events.jsonl"

    def write(self, event, **fields):
        with self.events.open("a", encoding="utf8") as out:
            out.write(json.dumps(dict(event=event, **fields), ensure_ascii=False, allow_nan=False) + "\n")

    def result(self, value):
        temp = self.path / "result.tmp"
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        temp.replace(self.path / "result.json")


class ReactAgent:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or AgentConfig()

    def run(self, environment, task, *, output, seed=42, knowledge="", metadata=None):
        log = Recorder(output)
        history = []
        rounds = calls = api_calls = 0
        started = elapsed_clock()
        timed_out = False
        context_limited = False
        log.write("run", task=task, seed=seed, benchmark=environment.benchmark,
                  config=asdict(self.config), knowledge=knowledge, metadata=metadata or {})
        try:
            current = environment.reset(task, seed)
            calls += current.environment_calls
            api_calls += current.api_calls
            instruction, interface = current.instruction, current.interface
            initial_observation = current.text
            log.write("reset", observation=asdict(current))
            while not (current.terminated or current.truncated) and rounds < self.config.max_rounds:
                if self.config.task_timeout_seconds and elapsed_clock()-started >= self.config.task_timeout_seconds:
                    timed_out = True
                    break
                try:
                    messages, budget = build_messages(instruction, interface, current, history,
                                                       knowledge, self.config, self.model.count_tokens, initial_observation)
                except ContextOverflow as exc:
                    context_limited = True
                    log.write("context_limit", round=rounds, message=str(exc), executed=False)
                    break
                log.write("request", round=rounds, messages=messages, budget=budget)
                response = self.model.generate(messages, max_tokens=self.config.output_tokens, seed=seed+rounds)
                # Truncated generations must not execute partial Python or commands.
                parsed = parse_action(response.text, self.config.allow_python_fence and environment.benchmark == "appworld")
                if response.finish_reason != "stop":
                    parsed.action, parsed.status = None, "incomplete_generation"
                log.write("decision", round=rounds, generation=asdict(response), parsed=asdict(parsed))
                rounds += 1
                if parsed.action is None:
                    history.append(dict(response=response.text, executed=False,
                        controller_feedback="Parser: " + parsed.status + ". Return exactly one complete nonempty <action> block."))
                    log.write("parse_error", round=rounds-1, status=parsed.status, executed=False)
                    continue
                # A decision record exists even if the native environment subsequently raises.
                current = environment.step(parsed.action)
                calls += current.environment_calls
                api_calls += current.api_calls
                log.write("transition", round=rounds-1, action=parsed.action,
                          executed=True, observation=asdict(current))
                history.append(dict(response=response.text, executed=True,
                    environment_observation=clip(current.text, self.config.max_output_chars)))
            metrics = environment.evaluate()  # Evaluator results are never sent back to the actor.
            result = dict(status="complete", decision_rounds=rounds, environment_calls=calls,
                api_calls=api_calls, terminated=current.terminated,
                truncated=context_limited or timed_out or current.truncated or (not current.terminated and rounds>=self.config.max_rounds),
                stop_reason="context_limit" if context_limited else "task_timeout" if timed_out else "environment_terminated" if current.terminated else
                    "environment_truncated" if current.truncated else "decision_budget",
                metrics=metrics)
            log.result(result)
            return result
        except Exception as exc:
            log.write("error", error_type=type(exc).__name__, message=str(exc))
            log.result(dict(status="error", error_type=type(exc).__name__, message=str(exc),
                            decision_rounds=rounds, environment_calls=calls, api_calls=api_calls))
            raise
        finally:
            environment.close()
