import subprocess,re
r=subprocess.run(['build/bin/qwecpp_chat','--model','/home/hzy/models/Qwen/Qwen3-0.6B','--parallel','tp','--devices','0,1','--max-new-tokens','1','--debug'],input='你好\n/exit\n',text=True,capture_output=True)
assert r.returncode==0,r.stderr
line=next(x for x in r.stderr.splitlines() if '[debug] turn=' in x)
a=dict(part.split('=') for part in line.split()[1:])
assert int(a['tp_transfer_calls'])==112,a
assert int(a['tp_transfer_bytes'])==112*int(a['input_tokens'])*1024*4,a
assert int(a['tp_p2p_calls'])+int(a['tp_host_calls'])==112,a
assert float(a['tp_transfer_ms'])>=float(a['tp_source_wait_ms'])>=0,a
print(line)
