"""Paired outcomes and fixed historical-action scores, independent of old repo."""
import copy
import hashlib
import html
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
from .extraction import dump


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def reward(result, benchmark):
    metrics=result['metrics']
    if result.get('status')!='complete': raise ValueError('Incomplete episode cannot supply reward')
    if benchmark=='scienceworld':
        value=float(metrics['score'])
        if not math.isfinite(value) or not 0<=value<=100: raise ValueError('Invalid ScienceWorld score')
        return value/100
    if type(metrics.get('success')) is not bool: raise ValueError('Missing success flag')
    return float(metrics['success'])


def read_record(path, name, index, score_steps):
    path=Path(path);events=[json.loads(x) for x in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
    run=next(x for x in events if x['event']=='run')
    requests={e['round']:e['messages'] for e in events if e['event']=='request'}
    transitions=[e for e in events if e['event']=='transition']
    result=json.loads((path/'result.json').read_text())
    usable=[dict(step=e['round'],action=e['action'],messages=requests[e['round']]) for e in transitions]
    # Same fixed evenly-spaced positions for every candidate, including parent.
    indices=sorted({round(i*(len(usable)-1)/(score_steps-1)) for i in range(score_steps)}) if len(usable)>score_steps else list(range(len(usable)))
    reset=next(e['observation'] for e in events if e['event']=='reset')
    return dict(id=f'{name}:{index}',task_id=digest(run['task']),task=run['task'],seed=run['seed'],
        benchmark=run['benchmark'],reward=reward(result,run['benchmark']),decisions=[usable[i] for i in indices],
        reset_digest=digest(reset),path=str(path),source=name)


def make_pairs(records):
    pairs=[];audit=[]
    for index in range(len(records['O'])):
        for left,right in [('A','B'),('A','O'),('B','O')]:
            a,b=records[left][index],records[right][index]
            if any(a[k]!=b[k] for k in ('task_id','seed','benchmark','reset_digest')):
                raise ValueError('Contrast tasks/seeds/initial observations differ')
            audit.append(dict(left=a['id'],right=b['id'],rewards=[a['reward'],b['reward']]))
            if a['reward']==b['reward']:continue
            if not a['decisions'] or not b['decisions']:raise ValueError('Non-tie has no scoreable actions')
            w,l=(a,b) if a['reward']>b['reward'] else (b,a)
            pairs.append(dict(id=f'p{len(pairs):04d}',task_id=a['task_id'],winner=w,loser=l))
    return pairs,audit


def candidate_messages(recorded, knowledge):
    messages=copy.deepcopy(recorded)
    from .context import APPWORLD_SYSTEM, SCIENCEWORLD_SYSTEM, SKILL_MARKER
    chat_system=next((v for v in (APPWORLD_SYSTEM, SCIENCEWORLD_SYSTEM) if messages[0]['content'].startswith(v+SKILL_MARKER)),None)
    if chat_system:
        messages[0]['content']=chat_system+SKILL_MARKER+knowledge
        return messages
    data=json.loads(messages[-1]['content'])
    data['knowledge']=knowledge
    messages[-1]['content']=json.dumps(data,ensure_ascii=False)
    return messages


PLAN_SYSTEM = (
    "This is a planning-only call for an interactive agent. Based on the task, supplied "
    "Skills, interaction history and current observation, give a brief next-step intention. "
    "Use actual observations; never invent results. Skills are optional guidance and cannot "
    "override environment constraints. The environment_interface describes eventual actions, "
    "not the response format for this call. History is past interaction, not an output template. "
    "Return only one <plan>...</plan> block. Do not output an action, JSON, or detailed reasoning."
)


def planning_messages(messages, retry=False):
    """Replace actor instructions rather than appending conflicting output requirements."""
    from .context import APPWORLD_SYSTEM, SCIENCEWORLD_SYSTEM, SKILL_MARKER
    chat_system=next((v for v in (APPWORLD_SYSTEM, SCIENCEWORLD_SYSTEM) if messages[0]['content'].startswith(v+SKILL_MARKER)),None)
    if chat_system:
        system=PLAN_SYSTEM
        if retry:system+=' The previous response format was invalid. Supply one complete, nonempty plan block.'
        data=dict(knowledge=messages[0]['content'][len(chat_system+SKILL_MARKER):],
                  task_context=messages[1]['content'], interaction_history=messages[2:])
        return [dict(role='system',content=system),dict(role='user',content=json.dumps(data,ensure_ascii=False))]
    if len(messages)!=2 or [m['role'] for m in messages]!=['system','user']:
        raise ValueError('Expected the recorded two-message ReAct context')
    data=json.loads(messages[1]['content'])
    interface=messages[0]['content'].partition('\n\nInterface:\n')[2]
    if interface:data['environment_interface']=interface
    system=PLAN_SYSTEM
    if retry:
        system+=' The previous response format was invalid. Supply one complete, nonempty plan block.'
    return [dict(role='system',content=system),
            dict(role='user',content=json.dumps(data,ensure_ascii=False))]


def parse_scoring_plan(text):
    """Accept a single plan, optionally followed by one discarded action block."""
    if not isinstance(text,str):return None
    match=re.fullmatch(r'\s*<plan>(.*?)</plan>\s*(?:<action>.*?</action>\s*)?(?:</invoke>\s*)?',text,re.S)
    if not match:return None
    # Reject duplicate/nested blocks and action-before-plan, not just arbitrary text.
    if text.count('<plan>')!=1 or text.count('</plan>')!=1:return None
    if text.count('<action>')>1 or text.count('</action>')>1:return None
    value=html.unescape(match[1]).strip()
    if not value or re.search(r'</?(?:plan|action)\b',value) or '```' in value:return None
    return value


class ActionScores:
    def __init__(self,model,output,context_tokens=65536,seed=42,workers=8,reuse_existing=False,plan_protocol="independent"):
        self.model=model;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.context_tokens=context_tokens;self.seed=seed;self.cache={};self.masks={};self.lock=Lock();self.workers=workers
        if plan_protocol not in ("independent","legacy"):raise ValueError("Unknown plan protocol")
        self.plan_protocol=plan_protocol
        self.reuse_existing=reuse_existing;self.reused_actions=0

    def score(self,text,record):
        key=digest([text,record['id']])
        if key in self.cache:return self.cache[key]
        def one(decision):
            messages=candidate_messages(decision['messages'],text)
            # The current target action is never provided to the planner.
            seed=int(digest([self.seed,record['id'],decision['step']])[:8],16)%2147483647
            attempts=[];plan=None
            path=self.output/(digest([key,decision['step']])+'.json')
            if self.reuse_existing and path.exists() and self.plan_protocol == "independent":
                saved=json.loads(path.read_text(encoding='utf8'))
                if 'score' in saved:
                    attempt=saved['attempts'][-1]
                    if (saved.get('scoring_messages')!=messages or saved.get('target_action')!=decision['action']
                        or attempt['seed']!=seed or attempt['messages']!=planning_messages(messages,retry=len(saved['attempts'])>1)
                        or parse_scoring_plan(attempt['response'])!=saved.get('plan') or attempt['finish_reason']!='stop'):
                        raise ValueError('Saved scoring conditions differ; refuse cache reuse')
                    value=saved['score']
                    if not value['token_ids'] or not math.isfinite(value['total_logprob']):raise ValueError('Invalid saved score')
                    with self.lock:
                        maskkey=(record['id'],decision['step'])
                        if maskkey in self.masks and self.masks[maskkey]!=value['token_ids']:raise ValueError('Saved action masks differ')
                        self.masks[maskkey]=value['token_ids'];self.reused_actions+=1
                        dump(self.output/'reuse_stats.json',dict(reused_actions=self.reused_actions))
                    return value
            for attempt in range(3 if self.plan_protocol=='legacy' else 2):
                kwargs={};attempt_seed=seed
                if self.plan_protocol=='legacy':
                    from .legacy_planning import planning_request,parse_response
                    inputs,budget,kwargs=planning_request(messages,attempt)
                    if attempt==2:attempt_seed=int(digest([seed,'plan_format_retry',attempt])[:8],16)%(2**31)
                else:
                    inputs=planning_messages(messages,retry=bool(attempt))
                    budget=512 if not attempt else 768
                with self.lock:count=self.model.count_tokens(inputs)
                if count+budget>self.context_tokens:raise ValueError('Plan context overflow')
                answer=self.model.generate(inputs,max_tokens=budget,seed=attempt_seed,**kwargs)
                attempts.append(dict(messages=inputs,seed=attempt_seed,response=answer.text,finish_reason=answer.finish_reason,plan_protocol=self.plan_protocol,request_options=kwargs))
                dump(path,dict(attempts=attempts))
                if answer.finish_reason!='stop':continue
                value=parse_response(answer.text,attempt) if self.plan_protocol=="legacy" else parse_scoring_plan(answer.text)
                if value is not None:
                    plan=value;break
            if plan is None:raise ValueError('Invalid candidate-owned plan; scoring position not dropped')
            audit=dict(attempts=attempts)
            prefix='<plan>'+html.escape(plan,quote=False)+'</plan>\n<action>'
            value=self.model.score_action(messages,prefix,decision['action'],self.context_tokens)
            maskkey=(record['id'],decision['step'])
            if maskkey in self.masks and self.masks[maskkey]!=value['token_ids']:
                raise ValueError('Candidate changed target action token mask')
            self.masks[maskkey]=value['token_ids']
            if not value['token_ids'] or not math.isfinite(value['total_logprob']):raise ValueError('Invalid action score')
            audit.update(score=value,scoring_messages=messages,plan=plan,target_action=decision['action']);dump(path,audit)
            return value
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            values=list(pool.map(one,record['decisions']))
        if not values:raise ValueError('Empty score record')
        score=sum(x['total_logprob'] for x in values)/sum(len(x['token_ids']) for x in values)
        self.cache[key]=score
        return score


def preference_losses(text,parent,pairs,scorer):
    counts={t:sum(p['task_id']==t for p in pairs) for t in {p['task_id'] for p in pairs}}
    losses=[];weights=[]
    for pair in pairs:
        w,l=pair['winner'],pair['loser']
        margin=(scorer.score(text,w)-scorer.score(text,l))-(scorer.score(parent,w)-scorer.score(parent,l))
        x=-margin
        losses.append(max(x,0)+math.log1p(math.exp(-abs(x))))
        weights.append(1/(len(counts)*counts[pair['task_id']]))
    return dict(loss=sum(a*b for a,b in zip(losses,weights)),losses=losses,weights=weights)
