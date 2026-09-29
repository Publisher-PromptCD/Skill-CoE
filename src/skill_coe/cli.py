import argparse
import json
from pathlib import Path
from .agent import ReactAgent
from .context import AgentConfig
from .environments import make_environment
from .model import ChatModel


def main():
    parser = argparse.ArgumentParser(description='Run one auditable ReAct episode.')
    parser.add_argument('--config', required=True, help='Portable JSON configuration')
    parser.add_argument('--task', required=True, help='Task JSON file')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--knowledge', help='Optional UTF-8 knowledge file; omitted means No-Skill')
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
    cfg, task = read(args.config), read(args.task)
    if Path(args.output).exists():
        parser.error('Output directory already exists')
    agent_cfg = AgentConfig(**cfg.get('agent', {}))
    model = ChatModel(**cfg['model'])
    environment = make_environment(cfg['benchmark'], cfg.get('environment', {}))
    knowledge = Path(args.knowledge).read_text(encoding='utf8') if args.knowledge else ''
    result = ReactAgent(model, agent_cfg).run(environment, task, output=args.output,
                                             seed=args.seed, knowledge=knowledge, metadata=cfg)
    # Save portable config (the model API key itself is never a config field).
    (Path(args.output)/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
