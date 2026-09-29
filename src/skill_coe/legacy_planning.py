"""Planning protocol ported from experiment_framework.methods.contrast_plans.
Only the model-client interface is adapted; no historical target enters generation.
"""
import json
import re

def parse_planning_json(content):
    obj=json.loads(content)
    if (not isinstance(obj,dict) or set(obj)!={'plan'} or not isinstance(obj['plan'],str)
            or not obj['plan'].strip() or len(obj['plan'])>2000):
        raise ValueError('Invalid planning-only response')
    plan=obj['plan'].strip()
    # JSON strings may quote the output protocol (e.g. "an <action> block").
    # Reject actual executable/tagged blocks, not mere mentions of tag names.
    if ('```' in plan or re.match(r'\s*</?(?:action|plan)\b',plan,re.I)
            or re.search(r'<(action|plan)\b[^>]*>.*?</\1\s*>',plan,re.S|re.I)):
        raise ValueError('Planning-only response contains an action or plan block')
    return plan

def parse_plan(text):
    # Accept harmless preamble, but never a plan produced after an action.
    prefix=text.split('<action',1)[0]
    matches=list(re.finditer(r'<plan>(.*?)</plan>',prefix,re.S))
    if len(matches)==1:
        plan=matches[0][1]
    else:
        # Observed format typo: a plan terminated by </action> before any action.
        match=re.fullmatch(r'\s*<plan>(.*?)</action>\s*',prefix,re.S)
        if match:
            plan=match[1]
        elif ('<action' in text and prefix.strip() and len(prefix.strip())<=2000
              and not any(token in prefix for token in ('<','>','```','{','}'))):
            # Only already-generated prose BEFORE the first action is eligible.
            # Never reconstruct a rationale from the generated action body.
            plan=prefix.strip()
        else:raise ValueError('Missing complete pre-action plan')
    if not plan.strip() or '<plan' in plan or '<action' in plan:
        raise ValueError('Ambiguous plan')
    return plan

def planning_request(messages, attempt):
    # Generate the candidate's plan from the same task, Skills, and history, but
    # never ask for an action in this planning-only call.
    from .coe_evidence import planning_messages
    inputs=planning_messages(messages)
    inputs[0]['content']=('This is a planning-only call for an interactive agent. '
        'Use the supplied task, Skills, interaction history and current observation. '
        'Return a JSON object with one field "plan": a brief next-step plan in one or two sentences. '
        'State the intended next step and relevant condition concisely. '
        'Do not output executable code, action blocks, or a detailed reasoning trace. '
        'The environment_interface describes eventual actions, not this response format. '
        'History is past interaction, not an output template.')
    kwargs={'response_format':{'type':'json_schema','json_schema':{'name':'next_step_plan','strict':True,'schema':{
        'type':'object','properties':{'plan':{'type':'string','minLength':1,'maxLength':2000}},
        'required':['plan'],'additionalProperties':False}}}}
    return inputs, (512 if attempt==0 else 768 if attempt==1 else 1024), kwargs

def parse_response(text,attempt):
    try:return parse_planning_json(text)
    except (ValueError,TypeError):return None
