import math
import re
from ..types import Observation

INTERFACE = "Complete the science task using one text command per action block.\nOBJ denotes an object name and LOC denotes a room name from the environment.\nAction syntax:\nopen OBJ / close OBJ: open or close a container\nactivate OBJ / deactivate OBJ: turn a device on or off\nconnect OBJ to OBJ / disconnect OBJ: connect or disconnect electrical components\nuse OBJ [on OBJ]: use an item or device\nlook around: describe the current room\nexamine OBJ: describe an object\nlook at OBJ: describe a container's contents\nread OBJ: read a note or book\nmove OBJ to OBJ: move an object into a container\npick up OBJ: take an object into inventory\npour OBJ into OBJ: pour a liquid\nmix OBJ: mix a container's contents\ngo to LOC: move to an accessible location\nteleport to LOC: teleport when enabled by the environment\nfocus on OBJ: signal intent on a task object\nwait: advance time by 10 steps\nwait1: advance time by one step\nCommands can fail; the returned observation is the actual execution result.\n"


class ScienceWorld:
    benchmark = 'scienceworld'

    def __init__(self, max_environment_steps=50, simplifications='', spatial_context=True, history_state_snapshots=False):
        self.limit, self.simplifications, self.spatial = max_environment_steps, simplifications, spatial_context
        self.history_state_snapshots = history_state_snapshots
        self.env = None

    def reset(self, task, seed):
        from scienceworld import ScienceWorldEnv
        self.close()
        self.steps, self.peak, self.rooms = 0, 0.0, []
        self.env = ScienceWorldEnv(envStepLimit=self.limit)
        # ScienceWorld variation, rather than a generic seed API, selects the world.
        self.env.load(task['task_name'], int(task['variation_id']),
                      simplificationStr=self.simplifications, generateGoldPath=False)
        text, self.info = self.env.reset()
        result = self._observation(text)
        result.instruction = self.env.get_task_description()
        result.interface = "" if self.history_state_snapshots else INTERFACE
        enabled = self.env.get_simplifications_used()
        if 'teleportAction' in enabled:
            result.interface += ' Teleportation is enabled: teleport to LOC.'
        if self.simplifications == 'easy':
            facts = ['Environment preset: easy.']
            if 'openDoors' in enabled: facts.append('Doors are initially open.')
            if 'openContainers' in enabled: facts.append('Containers are initially open.')
            facts.append('Rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.')
            result.interface += ' '+ ' '.join(facts)
        result.diagnostics.update(simplifications=enabled, requested_seed=seed,
            seed_semantics='world selected by task_name and variation_id; no seed method invoked')
        return result

    def _observation(self, text, done=False, reward=None):
        score = float(self.info['score'])
        if not math.isfinite(score) or score > 100:
            raise ValueError('Invalid ScienceWorld score')
        self.score = score
        self.peak = max(self.peak, score)
        visible = {}
        if self.spatial:
            look = self.info.get('look', '')
            inv = self.info.get('inv', '')
            look = look if isinstance(look,str) else ''
            inv = inv if isinstance(inv,str) else ''
            room = re.search(r'\bcalled the\s+([^\n.,]+)', look.split('.',1)[0])
            name = room[1].strip() if room else None
            if name and name not in self.rooms:
                self.rooms.append(name)
            visible = dict(current_room=name, visited_rooms=list(self.rooms),
                           room_observation=look, inventory=inv)
        if self.history_state_snapshots and visible:
            text = str(text)+'\n\n[Environment state snapshot]\n'+(
                'Current room: '+(name or 'unavailable')+
                '\nVisited rooms (this episode): '+(', '.join(self.rooms) or 'unavailable')+
                '\nCurrent inventory:\n'+(inv or 'unavailable')+
                '\nCurrent room observation:\n'+(look or 'unavailable'))
        truncated = self.steps >= self.limit and score < 100 and score >= 0
        return Observation(str(text), score=score, reward=reward, success=score>=100,
            terminated=(bool(done) or score>=100) and not truncated, truncated=truncated,
            environment_calls=int(self.steps>0), visible=visible,
            diagnostics={'native_done':bool(done), 'native_step_limit':self.limit,
                         'submitted_actions':self.steps})

    def step(self, action):
        text, reward, done, self.info = self.env.step(action)
        self.steps += 1
        return self._observation(text, done, float(reward))

    def evaluate(self):
        return dict(success=self.peak>=100, metric='AS', score=self.peak,
                    terminal_score=self.score, peak_score=self.peak,
                    score_protocol='episode_peak', native_step_limit=self.limit)

    def close(self):
        if self.env is not None:
            env, self.env = self.env, None
            env.close()
