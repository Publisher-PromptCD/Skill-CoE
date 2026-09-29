from pathlib import Path
import json
from ..types import Observation

INTERFACE = (
    'Return executable Python in the action block. Variables persist across submissions. '
    'Use print(...) to observe values. Access APIs through the provided apis object. '
    'Documentation: print(apis.api_docs.show_app_descriptions()); '
    'print(apis.api_docs.show_api_descriptions(app_name="...")); '
    'print(apis.api_docs.show_api_doc(app_name="...", api_name="...")). '
    'Replace documentation placeholders with actual names. '
    'For an operation-only task, call apis.supervisor.complete_task() with no answer '
    'argument (or answer=None). For a question requiring an answer, call '
    'apis.supervisor.complete_task(answer=value): return only the requested value; '
    'numeric answers are numbers. If unable to finish, status=\'fail\' is available. '
    'Do not access hidden evaluators, '
    'internal databases, or solution files.'
)


class AppWorld:
    benchmark = 'appworld'

    def __init__(self, data_root, experiment_name, max_environment_steps=50,
                 max_api_calls_per_interaction=1000, code_timeout_seconds=60):
        self.root, self.name = data_root, experiment_name
        self.limit, self.api_limit = max_environment_steps, max_api_calls_per_interaction
        self.timeout = code_timeout_seconds
        self.world = None
        self.steps = 0

    def reset(self, task, seed):
        from appworld import AppWorld as NativeWorld
        from appworld.common.path_store import path_store
        self.close()
        self.steps = 0
        path_store.update_root(str(Path(self.root).resolve()))
        self.world = NativeWorld(task_id=task['task_id'], experiment_name=self.name,
            random_seed=seed, max_interactions=self.limit,
            max_api_calls_per_interaction=self.api_limit, timeout_seconds=self.timeout,
            load_ground_truth=True, ground_truth_mode='minimal')
        # Ground truth stays inside the native evaluator, never in public prompt fields.
        supervisor = {k:getattr(self.world.task.supervisor,k,'')
                      for k in ('first_name','last_name','email','phone_number')}
        return Observation('Python environment ready.\nSupervisor:\n'+json.dumps(supervisor,ensure_ascii=False), instruction=self.world.task.instruction,
                           interface=INTERFACE, visible={'supervisor':supervisor})

    def step(self, action):
        tracker = self.world.requester.request_tracker
        before = len(tracker.requests)
        tracker.max_num_requests = self.api_limit
        text = str(self.world.execute(action))
        self.steps += 1
        completed = bool(self.world.task_completed())
        return Observation(text, terminated=completed,
            truncated=not completed and self.steps>=self.limit,
            environment_calls=1, api_calls=len(tracker.requests)-before)

    def evaluate(self):
        result = self.world.evaluate(suppress_errors=True)
        return dict(success=bool(result.success), metric='TGC')

    def close(self):
        if self.world is not None:
            world, self.world = self.world, None
            world.close()
