import subprocess,re,sys
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
conversation = '你好，请介绍一下自己。\n我刚才问了什么？\n/clear\n你好\n/exit\n'
baseline = subprocess.run(base + ['--kv-cache'], input=conversation, text=True, capture_output=True, check=True)
for route in [[], ['--pp-host-transfer']]:
    pd = subprocess.run(base + ['--parallel', 'pd', '--devices', '0,1'] + route,
                        input=conversation, text=True, capture_output=True, check=True)
    assert pd.stdout == baseline.stdout
    calls = [int(n) for n in re.findall(r'kv_transfer_calls=(\d+)', pd.stderr)]
    assert len(calls) == 3 and all(n > 0 for n in calls), pd.stderr
    if route:
        assert [int(n) for n in re.findall(r'kv_host_calls=(\d+)', pd.stderr)] == calls, pd.stderr
print('PD greedy chat equality, multi-turn and clear passed')
print('cache on/off exact greedy chat equality, multi-turn and clear passed')

if '--four-gpu' in sys.argv:
    for inner in ('pp', 'tp'):
        group = ['--devices', '0,1', '--parallel', inner, '--kv-cache']
        baseline = subprocess.run(base + group, input=conversation, text=True,
                                  capture_output=True, check=True)
        pd = subprocess.run(base + ['--devices', '0,1,2,3', '--parallel', 'pd-' + inner],
                            input=conversation, text=True, capture_output=True, check=True)
        assert pd.stdout == baseline.stdout, inner
        calls = [int(n) for n in re.findall(r'kv_transfer_calls=(\d+)', pd.stderr)]
        assert len(calls) == 3 and all(n > 0 for n in calls), pd.stderr
        print('PD-' + inner.upper() + ' multi-turn and clear passed')
