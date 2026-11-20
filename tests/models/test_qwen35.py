import subprocess,tempfile,pathlib,numpy as np,torch
from transformers import Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as refimpl
path='/home/hzy/models/Qwen/Qwen3.5-0.8B'
torch.backends.cuda.matmul.allow_tf32=False
model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval()
for layer in model.model.language_model.layers:
 if hasattr(layer,'linear_attn'):
  layer.linear_attn.causal_conv1d_fn=None
  layer.linear_attn.chunk_gated_delta_rule=refimpl.torch_chunk_gated_delta_rule
  norm=refimpl.Qwen3_5RMSNormGated(128,eps=1e-6).cuda()
  norm.weight.data.copy_(layer.linear_attn.norm.weight.data)
  layer.linear_attn.norm=norm
with tempfile.TemporaryDirectory() as d:
 for n in [1,4,16,64]:
  ids=([1,2,3,4]*16)[:n]
  p=subprocess.run(['build/bin/qwecpp_forward','--model',path,'--tokens',','.join(map(str,ids)),'--output',d+'/out'],capture_output=True,text=True)
  assert p.returncode==0,p.stderr
  actual=np.fromfile(d+'/out',dtype=np.float32).reshape(n,-1)
  with torch.no_grad():ref=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).logits[0].float().cpu().numpy()
  err=np.abs(actual-ref);print(n,'max_abs',float(err.max()),'mean_abs',float(err.mean()),'argmax',actual[-1].argmax(),ref[-1].argmax(),flush=True)
  np.testing.assert_allclose(actual,ref,atol=0.0003,rtol=0.0001)
  assert actual[-1].argmax()==ref[-1].argmax()
print('Qwen3.5 reference passed')

from transformers import AutoTokenizer
import re
model_dir=path
tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
history=[];expected=[]
for question in ['你好，请只回答一句话。','刚才我问了什么？']:
 history.append({'role':'user','content':question})
 ids=tok.apply_chat_template(history,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
 if not isinstance(ids,torch.Tensor):ids=ids['input_ids']
 with torch.no_grad():answer=model.generate(input_ids=ids.cuda(),max_new_tokens=8,do_sample=False,use_cache=False)
 text=tok.decode(answer[0,ids.shape[-1]:],skip_special_tokens=True)
 expected.append(text);history.append({'role':'assistant','content':text})
p=subprocess.run(['build/bin/qwecpp_chat','--model',model_dir,'--max-new-tokens','8'],input='你好，请只回答一句话。\n刚才我问了什么？\n/clear\n你好，请只回答一句话。\n/exit\n',capture_output=True,text=True)
assert p.returncode==0,p.stderr
actual=re.findall(r'助手> (.*?)(?=\n\[|\n你> )',p.stdout,re.S)
assert actual==[expected[0],expected[1],expected[0]],(actual,expected,p.stdout)
for extra in [['--kv-cache'],['--devices','0,1'],['--parallel','tp','--devices','0,1']]:
 p=subprocess.run(['build/bin/qwecpp_chat','--model',model_dir]+extra,input='/exit\n',capture_output=True,text=True)
 assert p.returncode!=0,extra
print('Qwen3.5 native chat/reference and unsupported-mode rejection passed')
