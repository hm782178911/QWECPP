import subprocess,re
exe='build/bin/qwecpp_chat'
r=subprocess.run([exe,'--model','/home/hzy/models/Qwen/Qwen3-0.6B','--max-new-tokens','4','--debug'],input='你好\n/clear\n你好\n/exit\n',capture_output=True,text=True)
assert r.returncode==0,r.stderr
assert '再见' in r.stdout
assert '[debug]' not in r.stdout
assert len(re.findall(r'\[debug\] turn=',r.stderr))==2,r.stderr
for key in ['ttft_ms','first_text_ms','total_ms','output_tokens','output_tok_s','tpot_ms']:
 assert key+'=' in r.stderr,key
print(r.stdout+r.stderr)
