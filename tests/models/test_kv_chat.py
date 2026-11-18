import subprocess,re
base=['build/bin/qwecpp_chat','--model','/home/hzy/models/Qwen/Qwen3-0.6B','--max-new-tokens','16','--max-context','512','--debug']
for extra in [[],['--devices','0,1'],['--devices','0,1','--parallel','tp']]:
 outputs=[]
 for flag in [[],['--kv-cache']]:
  p=subprocess.run(base+extra+flag,input='你好，请介绍一下自己。\n我刚才问了什么？\n/clear\n你好\n/exit\n',text=True,capture_output=True)
  assert p.returncode==0,p.stderr
  outputs.append(p.stdout)
  if flag:
   assert 'kv_cache_bytes=' in p.stderr and 'prefill_ms=' in p.stderr and 'decode_ms=' in p.stderr
 assert outputs[0]==outputs[1],(extra,outputs)
print('cache on/off exact greedy chat equality, multi-turn and clear passed')
