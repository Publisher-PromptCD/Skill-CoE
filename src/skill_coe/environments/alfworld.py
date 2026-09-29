import os
from pathlib import Path
from ..types import Observation


class ALFWorld:
    benchmark = "alfworld"

    def __init__(self, data_root, native_config_path, max_environment_steps=50):
        self.root = Path(data_root).resolve()
        self.config_path = Path(native_config_path)
        self.limit = max_environment_steps
        self.env = None
        self.success = False
        self.steps = 0

    def reset(self, task, seed):
        import yaml
        from alfworld.agents.environment import get_environment
        self.close()
        self.steps, self.success = 0, False
        os.environ['ALFWORLD_DATA'] = str(self.root)
        config = yaml.safe_load(os.path.expandvars(self.config_path.read_text(encoding='utf8')))
        game = (self.root / task['gamefile']).resolve()
        if self.root not in game.parents or not game.is_file():
            raise ValueError('Game file missing or outside data_root')
        config['general'].update(random_seed=seed, use_cuda=False, training_method='dqn')
        config['rl']['training']['max_nb_steps_per_episode'] = self.limit
        config['env']['domain_randomization'] = False
        for key in ('data_path', 'eval_id_data_path', 'eval_ood_data_path'):
            config['dataset'][key] = str(game.parent)
        split = {'train':'train', 'validation':'eval_in_distribution', 'test':'eval_out_of_distribution'}[task['split']]
        owner = get_environment('AlfredTWEnv')(config, train_eval=split)
        if [Path(p).resolve() for p in owner.game_files] != [game]:
            raise ValueError('Native task filter did not select exactly the requested game')
        self.env = owner.init_env(batch_size=1)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        observations, self.info = self.env.reset()
        return self._observation(observations[0], instruction=str(observations[0]))

    def _observation(self, text, instruction='', reward=None, done=False):
        actions = self.info.get('admissible_commands')
        if not actions or not isinstance(actions[0], (tuple, list)):
            raise ValueError('Missing official admissible commands')
        truncated = not self.success and self.steps >= self.limit
        return Observation(str(text), instruction=instruction,
            interface='Return one text command from the current admissible actions.',
            visible={'admissible_actions':list(actions[0])},
            success=self.success, reward=reward,
            terminated=(bool(done) or self.success) and not truncated, truncated=truncated,
            environment_calls=int(self.steps > 0), diagnostics={'native_done':bool(done),
                'native_step_limit':self.limit, 'submitted_actions':self.steps})

    def step(self, action):
        observations, rewards, dones, self.info = self.env.step([action])
        self.steps += 1
        if 'won' not in self.info:
            raise ValueError('Missing ALFWorld success signal; cannot infer success from reward')
        self.success = bool(self.info['won'][0])
        return self._observation(observations[0], reward=float(rewards[0]), done=dones[0])

    def evaluate(self):
        return dict(success=self.success, metric='SR')

    def close(self):
        if self.env is not None:
            env, self.env = self.env, None
            env.close()
