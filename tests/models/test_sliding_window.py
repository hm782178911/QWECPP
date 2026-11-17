import subprocess,re
base=['build/bin/qwecpp_chat','--max-context','256','--model','/home/hzy/models/Qwen/Qwen3-0.6B','--max-new-tokens','32','--debug']
text=('请详细介绍一下大语言模型的用途。\n'*12)+'/exit\n'
r=subprocess.run(base+['--sliding-window'],input=text,capture_output=True,text=True)
assert r.returncode==0,r.stderr
rows=[dict(p.split('=') for p in line.split()[1:]) for line in r.stderr.splitlines() if '[debug] turn=' in line]
assert len(rows)==12,(len(rows),r.stdout)
assert any(int(x['dropped_turns'])>0 for x in rows),r.stderr
assert all(int(x['input_tokens'])+32<=256 for x in rows)
r=subprocess.run(base+['--sliding-window'],input='x'*3000+'\n你好\n/exit\n',capture_output=True,text=True)
assert r.returncode==0 and '上下文超过' in r.stdout
assert sum('[debug] turn=' in x for x in r.stderr.splitlines())==1
print('sliding history, reserve, overflow recovery passed')
