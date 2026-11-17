import subprocess,sys,re
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
exe,path=sys.argv[1:3]
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
torch.backends.cuda.matmul.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval()
questions=['你好，请只回答一句话。','我刚才让你怎么回答？']
history=[];expected=[]
for question in questions:
    history.append({'role':'user','content':question})
    ids=tok.apply_chat_template(history,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
    # Recent Transformers returns BatchEncoding for return_tensors.
    if not isinstance(ids,torch.Tensor): ids=ids['input_ids']
    with torch.no_grad(): out=model.generate(input_ids=ids,max_new_tokens=16,do_sample=False,use_cache=False)
    answer=tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True)
    expected.append(answer);history.append({'role':'assistant','content':answer})
text='\n'.join(questions+['/clear',questions[0],'x'*3000,'/exit'])+'\n'
r=subprocess.run([exe,'--max-context','256','--model',path,'--max-new-tokens','16']+sys.argv[3:],input=text,capture_output=True,text=True,timeout=180)
assert r.returncode==0,r.stderr
actual=re.findall(r'助手> (.*?)(?=\n\[|\n你> )',r.stdout,re.S)
assert actual==[expected[0],expected[1],expected[0]],(actual,expected,r.stdout)
assert '已清空对话' in r.stdout and '上下文超过' in r.stdout and '再见' in r.stdout
print(r.stdout)
print('Multi-turn, greedy reference, clear, length limit and exit passed')
