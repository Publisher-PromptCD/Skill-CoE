"""Bounded, auditable views of long trajectories for offline Skill analysis.

Raw episode files are never changed. Every source step is inspected in an ordered
chunk; later stages receive evidence accounts with the original step numbers.
"""
import json
from dataclasses import replace

from .batch_learning import reflection_call
from .extraction import dump


CHUNK_PROMPT = """Describe only the observed task evidence in this consecutive
trajectory segment. Give the relevant action, actual feedback, state change or
error, and original step numbers. Preserve order and prerequisites. The final
outcome is context, not proof that every action in this segment was useful.
Do not propose Skills, invent missing steps, or replay routine output. Retain
short exact error/result details when they determine what happened.
"""

MERGE_PROMPT = """Combine these consecutive evidence accounts into one concise
account of the same trajectory. Keep the task requirements, observed outcome,
decisive actions and feedback, corrections, unmet requirements, and original
step numbers. Preserve uncertain or conflicting evidence as uncertain. Do not
invent missing transitions or turn an observed condition into a requirement.
The raw record remains available by trajectory ID; do not reproduce its text.
"""


def fits(model, payload, system, cfg, output_tokens, limit=None):
    messages=[dict(role='system',content=system),
              dict(role='user',content=json.dumps(payload,ensure_ascii=False))]
    ceiling=min(cfg.reflection_input_tokens,cfg.context_tokens-output_tokens)
    if limit is not None:ceiling=min(ceiling,limit)
    return model.count_tokens(messages)<=ceiling-512


def _fragments(step):
    """A single giant action/observation must not defeat ordered chunking."""
    fields=('action','feedback','public_state')
    pieces={key:[step[key][i:i+1600] for i in range(0,len(step[key]),1600)]
            if isinstance(step.get(key),str) and len(step[key])>1600 else [step.get(key)]
            for key in fields}
    n=max(map(len,pieces.values()))
    for i in range(n):
        part={k:v for k,v in step.items() if k not in fields}
        part['fragment']=f'{i+1}/{n}'
        for key,values in pieces.items():
            if i<len(values) and values[i] is not None:part[key]=values[i]
        yield part


def _groups(model, header, rows, prompt, cfg, output_tokens, limit):
    groups=[];current=[]
    for row in rows:
        trial=current+[row]
        if fits(model,dict(**header,items=trial),prompt,cfg,output_tokens,limit):
            current=trial;continue
        if not current:
            raise ValueError('One evidence fragment exceeds the analysis budget')
        groups.append(current);current=[row]
        if not fits(model,dict(**header,items=current),prompt,cfg,output_tokens,limit):
            raise ValueError('One evidence fragment exceeds the analysis budget')
    if current:groups.append(current)
    return groups


def account(model, view, path, cfg, seed):
    """Inspect every step once, then hierarchically condense long accounts."""
    token_cap=min(12000,cfg.reflection_input_tokens)
    output_tokens=min(3072,cfg.reflection_output_tokens)
    retry_tokens=min(4096,cfg.context_tokens-token_cap-512)
    header={k:view[k] for k in ('id','benchmark','instruction','outcome','stop_reason') if k in view}
    header['initial_observation']=view.get('initial_observation')
    rows=[]
    for step in view.get('steps',[]):
        selected={k:step[k] for k in ('step','action','feedback','public_state',
                  'score','terminated','truncated') if k in step and step[k] not in (None,'','{}',{})}
        if fits(model,dict(**header,items=[selected]),CHUNK_PROMPT,cfg,output_tokens,token_cap):
            rows.append(selected)
        else:
            rows.extend(_fragments(selected))
    if not rows:rows=[dict(note='No executed steps in saved trajectory')]
    groups=_groups(model,header,rows,CHUNK_PROMPT,cfg,output_tokens,token_cap)
    notes=[]
    stage=replace(cfg,reflection_input_tokens=token_cap,
                  reflection_output_tokens=output_tokens)
    for i,group in enumerate(groups):
        payload=dict(**header,items=group)
        dump(path.with_name(path.name+f'_chunk_{i}_input.json'),payload)
        notes.append(reflection_call(model,payload,path.with_name(path.name+f'_chunk_{i}'),
                     stage,seed+i*10,CHUNK_PROMPT,retry_output_tokens=retry_tokens))
    level=0
    while len(notes)>1:
        groups=_groups(model,header,notes,MERGE_PROMPT,cfg,output_tokens,token_cap)
        if len(groups)==len(notes):raise ValueError('Evidence accounts cannot be combined within budget')
        merged=[]
        for i,group in enumerate(groups):
            if len(group)==1:merged.append(group[0]);continue
            payload=dict(**header,items=group)
            dump(path.with_name(path.name+f'_merge_{level}_{i}_input.json'),payload)
            merged.append(reflection_call(model,payload,
                path.with_name(path.name+f'_merge_{level}_{i}'),stage,
                seed+100000+level*1000+i*10,MERGE_PROMPT,
                retry_output_tokens=retry_tokens))
        notes=merged;level+=1
    result=dict(trajectory_id=view['id'],benchmark=view.get('benchmark'),
                instruction=view.get('instruction'),outcome=view.get('outcome'),
                source_steps=[s.get('step') for s in view.get('steps',[])],
                evidence_account=notes[0],raw_record='Saved episode and trajectory view')
    dump(path.with_name(path.name+'_account.json'),result)
    return result
