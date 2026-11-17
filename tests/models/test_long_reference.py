import subprocess,tempfile,pathlib,sys,numpy as np,torch
from transformers import AutoModelForCausalLM
model=sys.argv[1];n=int(sys.argv[2]);ids=([9707,11,1879,0]*((n+3)//4))[:n]
with tempfile.TemporaryDirectory() as d:
 arrays=[]
 for args in [['--devices','0'],['--devices','0,1','--parallel','tp']]:
  out=d+'/out'
  subprocess.run(['build/bin/qwecpp_forward','--model',model,'--max-context','2048','--tokens',','.join(map(str,ids)),'--output',out]+args,check=True,stdout=subprocess.DEVNULL)
  arrays.append(np.fromfile(out,dtype=np.float32).reshape(n,-1)[-1].copy())
 torch.backends.cuda.matmul.allow_tf32=False
 m=AutoModelForCausalLM.from_pretrained(model,dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval()
 with torch.no_grad(): ref=m(torch.tensor([ids],device='cuda'),use_cache=False).logits[0,-1].cpu().numpy()
 for label,a in zip(['single','tp'],arrays):
  np.testing.assert_allclose(a,ref,atol=0.003,rtol=0.0001)
  assert a.argmax()==ref.argmax()
  print(model,n,label,'max_abs',float(np.max(np.abs(a-ref))),flush=True)
