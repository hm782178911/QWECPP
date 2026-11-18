import subprocess,json,time,pathlib,statistics,hashlib
r=pathlib.Path('/home/hzy/project/QWECPP');out=r/'benchmarks/kv-cache'/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
records=[]
prompts={'short':'请详细介绍大语言模型的用途。','long':'背景资料：'+('大语言模型可以用于文本生成、问答和代码辅助。'*12)+' 请详细解释这些用途。'}
for model in ['Qwen3-0.6B','Qwen3-8B']:
 for mode,args in [('single',['--devices','0']),('tp',['--parallel','tp','--devices','0,1'])]:
  for cache in [False,True]:
   argv=[str(r/'build/bin/qwecpp_chat'),'--model','/home/hzy/models/Qwen/'+model,'--max-context','512','--max-new-tokens','32','--debug']+args+(['--kv-cache'] if cache else [])
   text=''.join((prompt+'\n/clear\n')*4 for prompt in prompts.values())+'/exit\n'
   result=subprocess.run(argv,input=text,text=True,capture_output=True,check=True)
   name=f'{model}-{mode}-cache{int(cache)}'
   (out/(name+'.stdout.txt')).write_text(result.stdout);(out/(name+'.stderr.txt')).write_text(result.stderr)
   rows=[{k:float(v) for k,v in (part.split('=') for part in line.split()[1:])} for line in result.stderr.splitlines() if line.startswith('[debug] turn=')]
   assert len(rows)==8,result.stderr
   for i,case in enumerate(prompts):
    runs=rows[i*4+1:i*4+4];item={'model':model,'mode':mode,'cache':cache,'case':case,'prompt':prompts[case],'argv':argv,'warmup':rows[i*4],'runs':runs,'median':{k:statistics.median(x[k] for x in runs) for k in runs[0] if k!='turn'}}
    records.append(item);print(model,mode,cache,case,json.dumps({k:item['median'][k] for k in ['input_tokens','output_tokens','ttft_ms','tpot_ms','output_tok_s','kv_cache_bytes']}),flush=True)
   (out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
meta={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'binary_sha256':hashlib.sha256((r/'build/bin/qwecpp_chat').read_bytes()).hexdigest(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv'],text=True),'method':'same binary; debug on both; max-context 512; max-new 32; each case 1 warmup then 3 runs; per-turn reset; not clock-locked or GPU-isolated'}
(out/'environment.json').write_text(json.dumps(meta,indent=2));print('ARCHIVE',out,flush=True)
