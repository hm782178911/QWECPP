import subprocess,time,json,statistics,os
from transformers import AutoTokenizer
root='/home/hzy/project/QWECPP'; model='/home/hzy/models/Qwen/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
question='请介绍一下什么是大语言模型，并说明它的用途。'
def read_until(p,marker):
 data=b''
 while not data.endswith(marker):
  b=p.stdout.read(1)
  if not b: raise RuntimeError(data.decode(errors='replace'))
  data+=b
 return data
results={}
for name,args in [('single',['--devices','0']),('pp',['--devices','0,1']),('tp',['--devices','0,1','--parallel','tp'])]:
 start=time.perf_counter()
 p=subprocess.Popen([root+'/build/bin/qwecpp_chat','--model',model,'--max-new-tokens','32']+args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 read_until(p,'你> '.encode());load=time.perf_counter()-start
 rows=[]
 for i in range(6):
  p.stdin.write((question+'\n').encode());p.stdin.flush();start=time.perf_counter()
  read_until(p,'助手> '.encode())
  first=p.stdout.read(1);ttft=time.perf_counter()-start
  rest=read_until(p,'你> '.encode());elapsed=time.perf_counter()-start
  reply=(first+rest).decode().split('\n[')[0].split('\n你> ')[0].rstrip('\n')
  tokens=len(tok.encode(reply,add_special_tokens=False))
  row={'ttft_ms':ttft*1000,'total_ms':elapsed*1000,'output_tokens_reencoded':tokens,'output_tok_s':tokens/elapsed,'post_first_ms_per_token':(elapsed-ttft)*1000/(tokens-1) if tokens>1 else None}
  if i: rows.append(row)
  p.stdin.write(b'/clear\n');p.stdin.flush();read_until(p,'你> '.encode())
 p.stdin.write(b'/exit\n');p.stdin.flush();p.wait()
 results[name]={'load_seconds':load,'runs':rows,'median':{k:statistics.median(r[k] for r in rows) for k in rows[0]}}
 print(name,json.dumps(results[name]),flush=True)
open(root+'/docs/tp-chat-benchmark.json','w').write(json.dumps({'prompt':question,'max_new_tokens':32,'warmups':1,'repeats':5,'results':results},ensure_ascii=False,indent=2))
