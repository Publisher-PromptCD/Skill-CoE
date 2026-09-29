"""v6 grouped experience + contrast-guided incremental skill evolution."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,dataclass,replace
from pathlib import Path
from .batch_learning import BatchConfig,empty_state,render,validate_state,environmental_view,reflection_call
from .extraction import ExtractionConfig,read_episode,request,dump
from .grouped_probe import run as grouped_analysis,analysis_protocol
from .coe_skills import generation_prompt,apply_patch
from .recombination import build_plan
from .coe_evidence import digest,read_record,make_pairs,ActionScores,preference_losses,reward

@dataclass
class CoEConfig:
    plan_protocol:str="independent"
    inner_proposals:int=3
    evidence_per_role:int=2
    score_steps:int=16
    score_workers:int=8
    success_replay:int=2
    success_cache:int=24
    seed:int=42
    recombination_output_tokens:int=6144
    recombination_retry_tokens:int=8192
    recombination_workers:int=4
    def __post_init__(self):
        if self.plan_protocol not in ("independent","legacy"):raise ValueError("Invalid plan protocol")
        for name in ('inner_proposals','evidence_per_role','score_steps','score_workers','success_replay','success_cache','recombination_output_tokens','recombination_retry_tokens','recombination_workers'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('Invalid '+name)
        if self.recombination_retry_tokens<self.recombination_output_tokens:raise ValueError('Retry budget cannot shrink')
        if self.score_steps<2:raise ValueError('score_steps must be >=2')


def new_candidate(state,parent):
    return bool(state['entries']) and [r['content'] for r in state['entries']]!=[r['content'] for r in parent['entries']]


def generate(model,parent,material,ids,path,cfg,seed):
    channel=material.get('channel')
    prompt=generation_prompt(channel)
    if channel=='B' and parent['entries']:raise ValueError('Reconstruction cannot receive prior Skills')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    data=dict(current_skills=parent,evidence=material,allowed_evidence_ids=sorted(ids))
    def validate(obj):
        candidate=apply_patch(parent,obj,ids)
        if model.count_tokens([dict(role='user',content=render(candidate))])>cfg.playbook_tokens:
            raise ValueError('Candidate exceeds playbook budget')
        return obj
    delta=request(model,prompt,data,validate,path/'edit',cfg.curation_input_tokens,cfg.curation_output_tokens,
        ExtractionConfig(context_tokens=cfg.context_tokens,format_attempts=cfg.format_attempts),seed)
    state=apply_patch(parent,delta,ids);dump(path/'state.json',state)
    (path/'playbook.md').write_text(render(state),encoding='utf8')
    return state


def select_new(archive,parent):
    eligible=[x for x in archive if new_candidate(x['state'],parent)]
    if not eligible:raise ValueError('No effective new candidate; parent is not silently redeployed')
    return min(eligible,key=lambda x:x['loss'])


def recombination_pool(archive):
    """Only generated Skills may be the subject of recombination."""
    pool=[x for x in archive if x['name']!='O']
    if not pool:
        raise ValueError('No generated Skills available for recombination')
    return pool


def sample(items,weights,n,rng):
    remaining=[(x,w) for x,w in zip(items,weights) if w>0];chosen=[]
    while remaining and len(chosen)<n:
        index=rng.choices(range(len(remaining)),weights=[w for _,w in remaining])[0]
        chosen.append(remaining.pop(index)[0])
    return chosen


def evidence(pair):
    def view(record):return environmental_view(read_episode(record['path'],record['id']),4000)
    return dict(id=pair['id'],task_id=pair['task_id'],winner_source=pair['winner']['source'],
        loser_source=pair['loser']['source'],winner=view(pair['winner']),loser=view(pair['loser']))


def optimize(model,parent,channels,pairs,successes,out,bc,cc,scorer):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not pairs:
        chosen=next((channels[k] for k in ('A','B') if new_candidate(channels[k],parent)),None)
        if chosen is None:raise ValueError('No preference and no effective native update')
        dump(out/'selection.json',dict(reason='no_pair_native_update',channel='A' if chosen is channels['A'] else 'B'))
        return chosen
    archive=[]
    def add(name,state):
        if any(render(x['state'])==render(state) for x in archive):return next(x for x in archive if render(x['state'])==render(state))
        row=dict(name=name,state=state,**preference_losses(render(state),render(parent),pairs,scorer))
        archive.append(row);dump(out/'archive.json',archive);return row
    for name,state in [('O',parent),*channels.items()]:add(name,state)
    subject=min(recombination_pool(archive),key=lambda x:x['loss']);rng=random.Random(cc.seed);previous=None
    details={p['id']:evidence(p) for p in pairs};iterations=[]
    for i in range(cc.inner_proposals):
        others=[a for a in archive if a is not subject]
        def gain(a):return sum(w*max(x-y,0) for w,x,y in zip(subject['weights'],subject['losses'],a['losses']))
        donor=max(others,key=gain) if others else subject
        absorb_weights=[w*max(x-y,0) for w,x,y in zip(subject['weights'],subject['losses'],donor['losses'])]
        # No complementary donor: investigate subject's weakest evidence without asserting donor superiority.
        complementary=any(absorb_weights)
        if not complementary:absorb_weights=[w*l for w,l in zip(subject['weights'],subject['losses'])]
        retain_weights=[w*max(y-x,0) for w,x,y in zip(subject['weights'],subject['losses'],donor['losses'])]
        roles=dict(ABSORB=sample(pairs,absorb_weights,cc.evidence_per_role,rng),
                   RETAIN=sample(pairs,retain_weights,cc.evidence_per_role,rng))
        plan,selected_ids=build_plan(model,subject,donor,roles,details,successes,previous,
                                     out,bc,cc,i,complementary)
        state=generate(model,subject['state'],dict(modification_plan=plan),selected_ids,out/f'C{i+1}',bc,cc.seed+3000+i)
        candidate=add(f'C{i+1}',state)
        changes=[dict(id=p['id'],old=a,new=b,direction='improved' if b<a else 'worsened' if b>a else 'unchanged')
                 for p,a,b in zip(pairs,subject['losses'],candidate['losses'])]
        selected_changes=sorted(changes,key=lambda x:abs(x['new']-x['old']),reverse=True)[:2*cc.evidence_per_role]
        previous=dict(operations=json.loads((out/f'C{i+1}/edit.json').read_text()),changes=changes,
                      cases=[details[x['id']] for x in selected_changes])
        iterations.append(dict(subject=subject['name'],reference=donor['name'],candidate=candidate['name'],
            absorb=[p['id'] for p in roles['ABSORB']],retain=[p['id'] for p in roles['RETAIN']],pair_changes=changes))
        dump(out/'iterations.json',iterations)
        if candidate['loss']<subject['loss'] and new_candidate(candidate['state'],parent):break
    chosen=select_new(archive,parent)
    dump(out/'selection.json',dict(name=chosen['name'],loss=chosen['loss'],reason='new_candidate_min_loss'))
    return chosen['state']


def same_execution_config(saved, expected, original_directory):
    saved,expected=copy.deepcopy(saved),copy.deepcopy(expected)
    # Service endpoints and per-episode output names do not change model behavior.
    for cfg in (saved,expected):
        for key in ('base_url','tokenizer_url'):cfg.get('model',{}).pop(key,None)
    if expected.get('benchmark')=='appworld':
        base=expected['environment']['experiment_name']
        actual=saved['environment']['experiment_name']
        if actual!=base+'_'+digest(str(Path(original_directory).resolve()))[:16]:return False
        saved['environment']['experiment_name']=base
    return saved==expected


def reuse_analysis(source,runner,tasks,seeds,parent,out,cc):
    """Reuse a verified collection/analysis boundary, never old generated candidates."""
    source=Path(source)
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    if read(source/'parent.json')!=parent:raise ValueError('Resume parent differs')
    if hasattr(runner,'config') and not same_execution_config(read(source/'ordinary/config.json'),runner.config,source/'ordinary'):
        raise ValueError('Resume execution configuration differs')
    status=read(source/'analysis/status.json')
    benchmark=runner.config.get('benchmark') if hasattr(runner,'config') else None
    if read(source/'analysis/manifest.json').get('analysis_protocol')!=analysis_protocol(benchmark):
        raise ValueError('Resume analysis protocol differs; regenerate group analysis with current prompts')
    if status.get('status')!='complete' or status.get('stage')!='analysis_only':
        raise ValueError('Resume analysis incomplete')
    paths=sorted((source/'ordinary').glob('e[0-9][0-9][0-9]'))
    if len(paths)!=len(tasks):raise ValueError('Resume task count differs')
    for i,p in enumerate(paths):
        rec=read_record(p,'ordinary',i,cc.score_steps)
        if rec['task']!=tasks[i] or rec['seed']!=seeds[i]:raise ValueError('Resume task/seed differs')
    groups=read(source/'analysis/grouping.json')['groups']
    analyses=[read(source/f'analysis/group_{i:03d}_analysis.json') for i in range(len(groups))]
    if any(a['group']!=g or not a.get('analysis') for a,g in zip(analyses,groups)):
        raise ValueError('Resume group analysis differs')
    hashes={str(p.relative_to(source)):digest(read(p)) for p in (source/'analysis').glob('*.json')}
    for name in ('ordinary','views','analysis'):shutil.copytree(source/name,out/name)
    dump(out/'reuse.json',dict(source=str(source.resolve()),stage='after_group_analysis',analysis_hashes=hashes,
                               regenerated=['A','B','contrast/A','contrast/B','scores','search']))
    return [out/'ordinary'/p.name for p in paths],analyses


def reuse_ordinary(source,runner,tasks,seeds,parent,out,cc):
    """Resume a failed analysis without repeating completed environment rollouts."""
    source=Path(source)
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    if read(source/'parent.json')!=parent:
        raise ValueError('Resume parent differs')
    if hasattr(runner,'config') and not same_execution_config(
            read(source/'ordinary/config.json'),runner.config,source/'ordinary'):
        raise ValueError('Resume execution configuration differs')
    paths=sorted((source/'ordinary').glob('e[0-9][0-9][0-9]'))
    if len(paths)!=len(tasks):
        raise ValueError('Resume task count differs')
    for i,path in enumerate(paths):
        rec=read_record(path,'ordinary',i,cc.score_steps)
        if rec['task']!=tasks[i] or rec['seed']!=seeds[i]:
            raise ValueError('Resume task/seed differs')
    shutil.copytree(source/'ordinary',out/'ordinary')
    views=out/'views';views.mkdir()
    copied=[out/'ordinary'/p.name for p in paths]
    hashes={}
    for i,path in enumerate(copied):
        dump(views/f'e{i:03d}_view.json',environmental_view(read_episode(path,f'e{i:03d}'),8000))
        hashes[path.name]={name:hashlib.sha256((path/name).read_bytes()).hexdigest()
                           for name in ('events.jsonl','result.json')}
    dump(out/'reuse.json',dict(source=str(source.resolve()),stage='after_ordinary',
                                episode_hashes=hashes,regenerated=['views','analysis','A','B',
                                'contrast/A','contrast/B','scores','search']))
    return copied


def update(model,runner,tasks,seeds,parent,out,bc,cc,successes=None,scoring_model=None,reuse_from=None,reuse_prepared=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dump(out/'parent.json',parent)
    if reuse_from:
        source=Path(reuse_from)
        status_path=source/'analysis/status.json'
        status=json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else {}
        manifest_path=source/'analysis/manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {}
        benchmark=runner.config.get('benchmark') if hasattr(runner,'config') else None
        if status.get('status')=='complete' and status.get('stage')=='analysis_only' and \
                manifest.get('analysis_protocol')==analysis_protocol(benchmark):
            ordinary,analyses=reuse_analysis(source,runner,tasks,seeds,parent,out,cc)
        else:
            if reuse_prepared:raise ValueError('Prepared resume requires complete analysis')
            ordinary=reuse_ordinary(source,runner,tasks,seeds,parent,out,cc)
            analyses=grouped_analysis(model,out/'views',out/'analysis',bc,cc.seed,analyses_only=True)
    else:
        ordinary=runner(tasks,parent,out/'ordinary',seeds)
        views=out/'views';views.mkdir()
        for i,p in enumerate(ordinary):dump(views/f'e{i:03d}_view.json',environmental_view(read_episode(p,f'e{i:03d}'),8000))
        analyses=grouped_analysis(model,views,out/'analysis',bc,cc.seed,analyses_only=True)
    material=[dict(id=f'g{i:03d}',**a) for i,a in enumerate(analyses)];ids={x['id'] for x in material}
    # Shared trajectory interpretation; reconstruction never sees parent Skills.
    channels={}
    for name,base in [('A',parent),('B',empty_state())]:
        if reuse_prepared:
            source=Path(reuse_from)
            requests=sorted((source/name).glob('edit_attempt_*.request.json'))
            saved=json.loads(requests[-1].read_text(encoding='utf8'))
            # Format retries may append a repair instruction; the base prompt must match.
            if not saved['messages'][0]['content'].startswith(generation_prompt(name)):
                raise ValueError('Saved A/B generation prompt differs')
            candidate=json.loads((source/name/'state.json').read_text(encoding='utf8'));validate_state(candidate)
            delta=json.loads((source/name/'edit.json').read_text(encoding='utf8'))
            if candidate!=apply_patch(base,delta,ids):raise ValueError('Saved candidate does not match its edit')
            shutil.copytree(source/name,out/name);channels[name]=candidate
        else:
            channels[name]=generate(model,base,dict(group_analyses=material,channel=name),ids,out/name,bc,cc.seed+500+(name=='B'))
    records={}
    for name,state in [('O',parent),*channels.items()]:
        if name=='O':
            records[name]=[read_record(p,name,i,cc.score_steps) for i,p in enumerate(ordinary)]
            continue
        if reuse_prepared:
            source=Path(reuse_from)/'contrast'/name
            if (source/'knowledge.md').read_text(encoding='utf8')!=render(state):raise ValueError('Saved execution Skills differ')
            if hasattr(runner,'config') and not same_execution_config(json.loads((source/'config.json').read_text()),runner.config,source):
                raise ValueError('Saved contrast configuration differs')
            paths=sorted(source.glob('e[0-9][0-9][0-9]'))
            if len(paths)!=len(tasks):raise ValueError('Saved contrast incomplete')
            for j,p in enumerate(paths):
                rec=read_record(p,name,j,cc.score_steps)
                if rec['task']!=tasks[j] or rec['seed']!=seeds[j]:raise ValueError('Saved contrast task/seed differs')
            shutil.copytree(source,out/'contrast'/name)
            paths=[out/'contrast'/name/p.name for p in paths]
        else:
            paths=runner(tasks,state,out/'contrast'/name,seeds)
        records[name]=[read_record(p,name,i,cc.score_steps) for i,p in enumerate(paths)]
    dump(out/'parent_trajectory_reuse.json',dict(source='ordinary',paths=[str(p) for p in ordinary],
                                               additional_parent_rollouts=0))
    pairs,audit=make_pairs(records);dump(out/'pairs.json',dict(pairs=pairs,audit=audit))
    cache=list(successes or [])
    for i,p in enumerate(ordinary):
        rec=read_record(p,'ordinary',i,cc.score_steps)
        if rec['reward']==1:
            view=environmental_view(read_episode(p,'success_'+digest([str(out),i])[:12]),4000)
            cache=[x for x in cache if x['task_id']!=rec['task_id']]
            cache.append(dict(id=view['id'],task_id=rec['task_id'],trajectory=view))
    cache=cache[-cc.success_cache:];dump(out/'success_cache.json',cache)
    if reuse_prepared:
        shutil.copytree(Path(reuse_from)/'scores',out/'scores')
        dump(out/'reuse.json',dict(source=str(Path(reuse_from).resolve()),stage='before_recombination',
                                   reused=['ordinary','analysis','A','B','contrast/A','contrast/B','scores'],
                                   regenerated=['search']))
    selected=optimize(model,parent,channels,pairs,cache,out/'search',bc,cc,
                      ActionScores(scoring_model or model,out/'scores',bc.context_tokens,cc.seed,cc.score_workers,reuse_existing=reuse_prepared,plan_protocol=cc.plan_protocol))
    # Revision tracks deployed batches, independent of reconstruction's fresh IDs.
    selected=copy.deepcopy(selected);selected['revision']=parent['revision']+1
    dump(out/'state.json',selected);(out/'playbook.md').write_text(render(selected),encoding='utf8')
    dump(out/'status.json',dict(status='complete',pairs=len(pairs),revision=selected['revision']))
    return selected,cache


class ProcessRunner:
    """One native environment and tokenizer per process; eight concurrent episodes."""
    def __init__(self,config,workers=8):self.config=config;self.workers=workers
    def __call__(self,tasks,state,out,seeds):
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        execution=copy.deepcopy(self.config)
        if execution.get('benchmark')=='appworld':
            execution['environment']['experiment_name']+='_'+digest(str(out.resolve()))[:16]
        dump(out/'config.json',execution);(out/'knowledge.md').write_text(render(state),encoding='utf8')
        def one(item):
            i,task=item;dump(out/f'task_{i}.json',task);target=out/f'e{i:03d}'
            with (out/f'e{i:03d}.log').open('w',encoding='utf8') as log:
                result=subprocess.run([sys.executable,'-m','skill_coe.cli','--config',str(out/'config.json'),
                    '--task',str(out/f'task_{i}.json'),'--output',str(target),'--seed',str(seeds[i]),
                    '--knowledge',str(out/'knowledge.md')],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Episode failed: '+str(target))
            return target
        with ThreadPoolExecutor(max_workers=self.workers) as pool:return list(pool.map(one,enumerate(tasks)))


def evaluate(runner,tasks,state,out,seed):
    paths=runner(tasks,state,out,[int(digest([seed,t])[:8],16)%2147483647 for t in tasks])
    values=[]
    for p in paths:
        result=json.loads((p/'result.json').read_text());run=json.loads((p/'events.jsonl').read_text(encoding='utf8').splitlines()[0])
        values.append(reward(result,run['benchmark']))
    metric=sum(values)/len(values);dump(Path(out)/'summary.json',dict(mean_reward=metric,episodes=len(values)))
    return metric


def task_identity(task):
    if 'gamefile' in task:return digest(['alfworld',task['gamefile'].replace('\\','/')])
    if 'task_name' in task and 'variation_id' in task:return digest(['scienceworld',task['task_name'],int(task['variation_id'])])
    if 'task_id' in task:return digest(['appworld',task['task_id']])
    return digest(task)


def learn(model,runner,batches,validation,test,out,bc,cc,initial=None,scoring_model=None,
          initial_test=True,final_test=True,reuse_first_batch=None,resume_run=None):
    state=copy.deepcopy(initial if initial is not None else empty_state());validate_state(state)
    if not batches or not validation or (not test and (final_test or (initial_test and state['entries']))) or any(not b or len(b)>bc.batch_size for b in batches):raise ValueError('Empty/oversized datasets')
    sets=[{task_identity(t) for t in x} for x in ([t for b in batches for t in b],validation,test)]
    if any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]):raise ValueError('Training/validation/test overlap')
    if len(sets[1])!=len(validation) or len(sets[2])!=len(test) or any(len({task_identity(t) for t in b})!=len(b) for b in batches):
        raise ValueError('Duplicate tasks within a batch or evaluation set')
    if resume_run and reuse_first_batch:raise ValueError('Choose only one resume mode')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dump(out/'state_initial.json',state)
    dump(out/'protocol.json',dict(learning=asdict(bc),coe=asdict(cc),batches=batches,validation=validation,test=test,
                                  initial_test=initial_test,final_test=final_test))
    best=None;best_score=None;cache=[];curve=[];start=0
    try:
        if resume_run:
            source=Path(resume_run);read=lambda p:json.loads(p.read_text(encoding='utf8'))
            protocol=read(source/'protocol.json')
            for key,value in dict(batches=batches,validation=validation,test=test,learning=asdict(bc),
                                  initial_test=initial_test,final_test=final_test).items():
                if protocol.get(key)!=value:raise ValueError('Resume protocol differs: '+key)
            old_cc=CoEConfig(**protocol['coe'])
            allowed={'recombination_output_tokens','recombination_retry_tokens','recombination_workers'}
            if old_cc.plan_protocol!=cc.plan_protocol:
                if (source/'curve.json').exists() and read(source/'curve.json'):
                    raise ValueError('Cannot change scoring protocol after completed updates')
                allowed.add('plan_protocol')
            if any(asdict(old_cc)[k]!=v for k,v in asdict(cc).items() if k not in allowed):
                raise ValueError('Resume scoring/learning configuration differs')
            if read(source/'state_initial.json')!=state:raise ValueError('Resume initial state differs')
            curve=read(source/'curve.json') if (source/'curve.json').exists() else []
            start=len(curve)
            if start>=len(batches) or [r['batch'] for r in curve]!=list(range(start)):
                raise ValueError('Expected contiguous completed prefix and unfinished batch')
            for i,row in enumerate(curve):
                old=source/f'batch_{i:03d}'
                if read(old/'status.json')['status']!='complete':raise ValueError('Resume completed batch missing')
                if read(source/f'validation_{i:03d}/summary.json')['mean_reward']!=row['validation']:
                    raise ValueError('Resume validation differs')
                if read(old/'parent.json')!=state:raise ValueError('Resume parent chain differs')
                state=read(old/'state.json');validate_state(state)
                shutil.copytree(old,out/old.name)
                shutil.copytree(source/f'validation_{i:03d}',out/f'validation_{i:03d}')
            if start:
                cache=read(source/f'batch_{start-1:03d}/success_cache.json')
                winner=read(source/'best.json');best=winner['state'];best_score=winner['score'];validate_state(best)
                dump(out/'best.json',winner)
            elif (source/'best.json').exists():
                winner=read(source/'best.json');best=winner['state'];best_score=winner['score'];dump(out/'best.json',winner)
            dump(out/'curve.json',curve);dump(out/'current.json',state)
            dump(out/'resume.json',dict(source=str(source.resolve()),completed_batches=start,
                                       restart_stage='unfinished_batch',old_protocol=protocol))
        elif state['entries']:
            if initial_test:evaluate(runner,test,state,out/'test_initial',cc.seed)
            best_score=evaluate(runner,validation,state,out/'validation_initial',cc.seed);best=copy.deepcopy(state)
            dump(out/'best.json',dict(score=best_score,state=best,batch=-1))
        for i,tasks in enumerate(batches):
            if i<start:continue
            seeds=[int(digest([cc.seed,i,t])[:8],16)%2147483647 for t in tasks]
            old_batch=Path(resume_run)/f'batch_{i:03d}' if resume_run and i==start else None
            if old_batch and not all(
                (old_batch/'ordinary'/f'e{j:03d}'/'result.json').exists() and
                (old_batch/'ordinary'/f'e{j:03d}'/'events.jsonl').exists()
                for j in range(len(tasks))
            ):
                # An interrupted collection may contain only a subset of episodes.
                # Restart that unfinished batch; completed earlier batches stay copied.
                old_batch=None
            prepared=bool(old_batch and (old_batch/'pairs.json').exists()
                          and all((old_batch/name/'state.json').exists() for name in ('A','B')))
            dump(out/'status.json',dict(status='running',completed_batches=len(curve),active_batch=i,
                                       stage='recombination_resume' if prepared else 'learning'))
            state,cache=update(model,runner,tasks,seeds,state,out/f'batch_{i:03d}',bc,replace(cc,seed=cc.seed+i*10000),cache,scoring_model,
                               reuse_from=old_batch if old_batch else reuse_first_batch if i==0 else None,
                               reuse_prepared=prepared)
            dump(out/'current.json',state)
            score=evaluate(runner,validation,state,out/f'validation_{i:03d}',cc.seed)
            if best_score is None or score>best_score:
                best_score=score;best=copy.deepcopy(state);dump(out/'best.json',dict(score=score,state=best,batch=i))
            curve.append(dict(batch=i,validation=score,best_validation=best_score,revision=state['revision']));dump(out/'curve.json',curve)
            dump(out/'status.json',dict(status='running',completed_batches=i+1))
        final=evaluate(runner,test,best,out/'test_final',cc.seed) if final_test else None
        dump(out/'status.json',dict(status='complete',completed_batches=len(batches),best_validation=best_score,test=final,
                                   final_test_skipped=not final_test))
        return state,best
    except Exception as exc:
        dump(out/'status.json',dict(status='error',completed_batches=len(curve),error=str(exc)));raise


def main():
    from .model import ChatModel
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--state')
    p.add_argument('--reuse-first-batch',help='Reuse verified first-batch collection and analysis in a new output directory')
    p.add_argument('--resume-run',help='Copy completed batches and resume prepared candidates at recombination in a new output directory')
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text());data=json.loads(Path(a.data).read_text())
    initial=json.loads(Path(a.state).read_text()) if a.state else None
    execution={k:config[k] for k in ('benchmark','model','environment','agent') if k in config}
    learn(ChatModel(**config.get('learning_model',config['model'])),ProcessRunner(execution,config.get('workers',8)),
          data['batches'],data['validation'],data.get('test',[]),a.output,BatchConfig(**config.get('learning',{})),
          CoEConfig(**config.get('coe',{})),initial,ChatModel(**config['model']),
          reuse_first_batch=a.reuse_first_batch,resume_run=a.resume_run,**config.get('evaluation',{}))

if __name__=='__main__':main()
