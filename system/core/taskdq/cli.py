"""The single supported command line interface."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__


def main(argv=None):
    parser=argparse.ArgumentParser(prog='taskdq',description='Task-conditioned tabular error candidates and repair experiments')
    parser.add_argument('--version',action='version',version='taskdq '+__version__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['validate','prepare','run']:
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--output')
        if name=='run':p.add_argument('--resume',action='store_true')
    p=sub.add_parser('demo');p.add_argument('--output',default='runs/demo')
    p=sub.add_parser('summarize');p.add_argument('--run',required=True)
    p=sub.add_parser('apply');p.add_argument('--model',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--output',required=True);p.add_argument('--row-id')
    p=sub.add_parser('final-test');p.add_argument('--config',required=True);p.add_argument('--variant',required=True)
    p=sub.add_parser('e1',help='Frozen E1 fixed-pipeline experiment; no language-model calls')
    p.add_argument('--config',required=True);p.add_argument('--stage',choices=['all','verify','fit','test','report'],default='all')
    p=sub.add_parser('e2',help='Formal E2 preparation, individual frozen units and final evaluation')
    p.add_argument('--config',required=True);p.add_argument('--stage',required=True,
        choices=['prepare','run','fit-final','final-test','report'])
    p.add_argument('--dataset');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int)
    p=sub.add_parser('e2-risk-select',help='Experimental frozen-candidate selection from development OOF caches; no fitting')
    p.add_argument('--unit',required=True);p.add_argument('--pools',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',required=True,type=int)
    for name in ['export-sft','export-contexts']:
        p=sub.add_parser(name);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
        p.add_argument('--datasets',required=True,help='Comma-separated source dataset names')
    for name in ['train-sft','train-grpo']:
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='e2-risk-select':
            from .e2_risk import select_cached
            result=select_cached(args.unit,args.pools,args.output,args.seed)
        elif args.command=='e2':
            from . import e2
            result=e2.run(args.config,args.stage,args.dataset,args.seed,args.fold)
        elif args.command=='e1':
            from . import e1
            result=e1.run(args.config,args.stage)
        elif args.command in ['validate','prepare','run','demo','summarize','apply','final-test']:
            from .config import load_config
            from . import experiment
            if args.command=='demo':
                from .demo import create
                config=load_config(create(args.output));result=experiment.run(config)
                result={'status':'demo_complete','output_dir':config['output_dir'],'metric_rows':len(result),'research_claim':False}
            elif args.command=='summarize':result=experiment.summarize(args.run)
            elif args.command=='apply':result=experiment.apply(args.model,args.csv,args.output,args.row_id)
            elif args.command=='final-test':result=experiment.final_test(load_config(args.config),args.variant)
            else:
                config=load_config(args.config,args.output)
                if args.command=='validate':result={'valid':True,'schema_version':1,'mode':config['mode'],
                    'datasets':[d['name'] for d in config['datasets']],'variants':[v['name'] for v in config['variants']]}
                elif args.command=='prepare':result={'prepared_dir':str(experiment.prepare(config)),'test_evaluated':False}
                else:result={'status':'complete','output_dir':config['output_dir'],'metric_rows':len(experiment.run(config,args.resume))}
        else:
            from . import learning
            if args.command.startswith('export-'):
                result=learning.export(args.run,args.output,args.datasets.split(','),'contexts' if args.command=='export-contexts' else 'sft')
            else:
                config=learning.training_config(args.config)
                function=learning.train_sft if args.command=='train-sft' else learning.train_grpo
                result=function(config,args.validate_only)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print('taskdq: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
