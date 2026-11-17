import subprocess,json,time,hashlib,pathlib,statistics,platform,os,sys
r=pathlib.Path('/home/hzy/project/QWECPP');out=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else r/'benchmarks/baselines'/time.strftime('%Y%m%d-%H%M%S-no-kv');out.mkdir(parents=True,exist_ok=False)
def cmd(args):return subprocess.check_output(args,text=True,cwd=r).strip()
files=sorted(p for folder in ['src','include','kernels','tools','third_party'] for p in (r/folder).rglob('*') if p.is_file())+[r/'CMakeLists.txt']
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
meta={'baseline':'no-kv-cache-two-gpu-tp','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':cmd(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,power.limit,temperature.gpu,utilization.gpu','--format=csv']),'nvcc':cmd(['/usr/local/cuda/bin/nvcc','--version']),'platform':platform.platform(),'cpu':cmd(['lscpu']),'git_status':cmd(['git','status','--short']),'source_hashes':hashes,'binary_sha256':hashlib.sha256((r/'build/bin/qwecpp_chat').read_bytes()).hexdigest(),'model_config':json.loads(pathlib.Path('/home/hzy/models/Qwen/Qwen3-0.6B/config.json').read_text()),'model_weight_sha256':hashlib.sha256(pathlib.Path('/home/hzy/models/Qwen/Qwen3-0.6B/model.safetensors').read_bytes()).hexdigest(),'method':{'warmups':1,'repeats':5,'max_new_tokens':32,'clear_between_requests':True,'clock':'steady_clock in chat --debug','timing':'tokenization through generation and output; excludes load; no KV cache','order':'single,pp,tp; sequential; not GPU-isolated'}}
(out/'environment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
prompts={'short':'请介绍一下什么是大语言模型，并说明它的用途。','long':'背景资料：'+('大语言模型可以用于文本生成、问答和代码辅助。'*12)+' 请简要总结上述资料。'}
records=[]
for name,args in [('single',['--devices','0']),('pp',['--devices','0,1']),('tp',['--devices','0,1','--parallel','tp'])]:
 for case,prompt in prompts.items():
  script=''.join(prompt+'\n/clear\n' for _ in range(6))+'/exit\n'
  argv=[str(r/'build/bin/qwecpp_chat'),'--model','/home/hzy/models/Qwen/Qwen3-0.6B','--max-context','256','--max-new-tokens','32','--debug']+args
  run=subprocess.run(argv,input=script,text=True,capture_output=True,check=True)
  (out/(name+'-'+case+'.stdout.txt')).write_text(run.stdout)
  (out/(name+'-'+case+'.stderr.txt')).write_text(run.stderr)
  rows=[]
  for line in run.stderr.splitlines():
   if line.startswith('[debug] turn='):
    row={k:float(v) for k,v in (part.split('=') for part in line.split()[1:])};rows.append(row)
  assert len(rows)==6,(case,run.stdout,run.stderr)
  item={'mode':name,'case':case,'prompt':prompt,'argv':argv,'warmup':rows[0],'runs':rows[1:]}
  item['median']={k:statistics.median(x[k] for x in rows[1:]) for k in rows[0] if k!='turn'}
  records.append(item);print(name,case,json.dumps(item['median']),flush=True)
(out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
lines=['# 无 KV Cache 性能基线 — 2026-11-17','','Qwen3-0.6B，FP32，两张 RTX PRO 6000 Blackwell Max-Q。每组独立进程加载，预热 1 轮、测量 5 轮，每轮清空历史，最多输出 32 token。下表为中位数，毫秒单位。长短输入的实际 token 数见表。首轮冷启动与逐次数据保留在 results.json；完整 stdout/stderr 和环境、源文件哈希保存在同目录。','','|模式|输入|输入 token|输出 token|TTFT ms|首文本 ms|总耗时 ms|token/s|TPOT ms|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in records:
 m=x['median'];lines.append('|'+x['mode']+'|'+x['case']+'|'+ '|'.join(f'{m[k]:.3f}' for k in ['input_tokens','output_tokens','ttft_ms','first_text_ms','total_ms','output_tok_s','tpot_ms'])+'|')
lines+=['','## 后续比较口径','','固定模型权重、输入 token、输出 token 数、dtype、设备顺序和预热/重复次数；每次优化保存新目录，保留本目录。分别报告 TTFT、TPOT、总耗时及吞吐相对本基线的变化。输出提前 EOS 或 token 数变化时标记为不可直接比较。KV Cache 优化后应另测 prefill/decode，并保留这些端到端指标。','','本次未锁定 GPU 时钟，也未隔离其他 GPU 进程；小样本结果用于工程基线，不代表严格实验结论。源码已有大量未提交内容，因此以 environment.json 的逐文件 SHA256 为版本依据。每组首次请求含惰性初始化，仅列入 warmup，不纳入中位数。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
