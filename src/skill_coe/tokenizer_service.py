"""Loopback-only tokenizer sidecar, run in the model environment, never AppWorld."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .model import ChatModel


def prepare(model, payload):
    if payload['model'] != model.name or payload.get('template_kwargs', {}) != model.template_kwargs:
        raise ValueError('Tokenizer model/template settings differ')
    messages=payload['messages']
    result=dict(model=model.name)
    if 'action' in payload:
        chat,ids,mask=model._local_score_tokens(messages,payload['prefix'],payload['action'])
        result.update(chat=chat,ids=ids,mask=mask,count=len(chat))
    else:
        result['count']=model.count_tokens(messages)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args()
    with open(a.config) as f:config=json.load(f)['model']
    config.pop('tokenizer_url',None)
    model=ChatModel(**config)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,obj):
            data=json.dumps(obj).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200 if self.path=='/health' else 404,{'model':model.name})
        def do_POST(self):
            if self.path!='/prepare':return self.reply(404,{'error':'Unknown route'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16*1024*1024:raise ValueError('Invalid payload size')
                self.reply(200,prepare(model,json.loads(self.rfile.read(size))))
            except (ValueError,KeyError,TypeError) as exc:self.reply(400,{'error':str(exc)})
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
