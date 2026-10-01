"""OpenAI-compatible local inference, with caller-supplied tokenizer chat template."""
from .failures import GenerationError
from .context import ContextOverflow
import json
import os
from threading import RLock
from urllib.request import Request, urlopen
from .types import Generation


class ChatModel:
    def __init__(self, *, base_url, model, tokenizer_path=None, api_key_env="MODEL_API_KEY",
                 temperature=0.7, timeout=600, template_kwargs=None, tokenizer_url=None,
                 top_p=None, top_k=None, presence_penalty=None):
        self.tokenizer_url = tokenizer_url.rstrip('/') if tokenizer_url else None
        if not self.tokenizer_url:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        self.tokenizer_lock = RLock()
        self.template_kwargs = template_kwargs or {}
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.name, self.key_env = model, api_key_env
        self.temperature, self.timeout = temperature, timeout
        self.sampling = {k:v for k,v in dict(top_p=top_p, top_k=top_k,
                         presence_penalty=presence_penalty).items() if v is not None}

    def count_tokens(self, messages):
        if getattr(self, 'tokenizer_url', None):
            return self._prepare(messages)['count']
        with self.tokenizer_lock:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, **self.template_kwargs)
        if hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
        if not isinstance(tokens, list) or any(not isinstance(t, int) for t in tokens):
            raise ValueError("Tokenizer must return a flat token sequence")
        return len(tokens)

    def _prepare(self, messages, **fields):
        payload=dict(model=self.name, messages=messages, template_kwargs=self.template_kwargs, **fields)
        req=Request(self.tokenizer_url+'/prepare', data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=self.timeout) as response:
            result=json.load(response)
        if result.get('model') != self.name:
            raise ValueError('Tokenizer service model differs')
        return result

    def generate(self, messages, *, max_tokens, seed, response_format=None):
        payload = dict(model=self.name, messages=messages, temperature=self.temperature,
                       max_tokens=max_tokens, seed=seed)
        payload.update(getattr(self, 'sampling', {}))
        if response_format is not None:
            payload['response_format'] = response_format
        if self.template_kwargs:
            payload['chat_template_kwargs'] = self.template_kwargs
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        choice = result['choices'][0]
        text = choice['message']['content']
        if not isinstance(text, str):
            raise GenerationError("Model returned no text content")
        return Generation(text, choice.get('finish_reason', 'unknown'), result.get('usage', {}))

    def score_action(self, messages, prefix, action, context_tokens=65536):
        """Teacher-force an action body; plan and markup are conditioning only."""
        import math
        raw = prefix + action + '</action>'
        if getattr(self, 'tokenizer_url', None):
            prepared=self._prepare(messages, prefix=prefix, action=action)
            chat,ids,mask=prepared['chat'],prepared['ids'],prepared['mask']
        else:
            chat,ids,mask=self._local_score_tokens(messages,prefix,action)
        if len(chat)+len(ids)+1 > context_tokens:
            raise ContextOverflow('Scoring exceeds context')
        body = dict(model=self.name, prompt=chat+ids, max_tokens=1, temperature=0.,
                    echo=True, logprobs=1, prompt_logprobs=1, add_special_tokens=False)
        headers={'Content-Type':'application/json'}
        if os.environ.get(self.key_env): headers['Authorization']='Bearer '+os.environ[self.key_env]
        req=Request(self.url.rsplit('/chat/completions',1)[0]+'/completions',data=json.dumps(body).encode(),headers=headers)
        with urlopen(req,timeout=self.timeout) as response: result=json.load(response)
        values=(result['choices'][0].get('logprobs') or {}).get('token_logprobs') or []
        if len(values)<len(chat)+len(ids): raise ValueError('Missing forced token probabilities')
        selected=[values[len(chat)+i] for i in mask]
        if any(v is None or not math.isfinite(float(v)) or float(v)>1e-5 for v in selected):
            raise ValueError('Invalid forced token probabilities')
        return dict(total_logprob=sum(selected),token_ids=[ids[i] for i in mask],token_mask=mask)

    def _local_score_tokens(self,messages,prefix,action):
        raw=prefix+action+'</action>'
        with self.tokenizer_lock:
            encoded = self.tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            mask = [i for i,(a,b) in enumerate(offsets) if a < len(prefix)+len(action) and b > len(prefix)]
            if not action or not mask:
                raise ValueError('Empty action score mask')
            chat = self.tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, **self.template_kwargs)
            if hasattr(chat, 'keys'): chat = chat['input_ids']
        return chat,ids,mask
