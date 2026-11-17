import pathlib,subprocess,sys,tempfile,json
import numpy as np
exe,model=sys.argv[1:3]
with tempfile.TemporaryDirectory() as tmp:
 def run(ids,args):
  out=pathlib.Path(tmp)/'logits'
  p=subprocess.run([exe,'--model',model,'--tokens',','.join(map(str,ids)),'--output',str(out)]+args,capture_output=True,text=True)
  assert p.returncode==0,p.stderr
  return np.fromfile(out,dtype=np.float32).reshape(len(ids),-1)
 for n in ([int(sys.argv[3])] if len(sys.argv)>3 else [1,4,32,256]):
  ids=([9707,11,1879,0]*64)[:n]
  ref=run(ids,['--devices','0'])
  for args in [['--devices','0,1'],['--devices','0,1','--parallel','tp'],['--devices','1,0','--parallel','tp']]:
   actual=run(ids,args)
   np.testing.assert_allclose(actual,ref,atol=2e-3,rtol=1e-4)
   assert actual[-1].argmax()==ref[-1].argmax()
   print(n,args,'max_abs',float(np.max(np.abs(actual-ref))),flush=True)
 bad=pathlib.Path(tmp)/'bad';bad.mkdir()
 (bad/'config.json').write_bytes((pathlib.Path(model)/'config.json').read_bytes())
 for index in [{},{'weight_map':{}},{'weight_map':{'model.embed_tokens.weight':'../bad.safetensors'}},{'weight_map':{'model.embed_tokens.weight':'missing.safetensors'}},{'weight_map':{'wrong':'missing.safetensors'}}]:
  (bad/'model.safetensors.index.json').write_text(json.dumps(index))
  p=subprocess.run([exe,'--model',str(bad),'--tokens','1'],capture_output=True)
  assert p.returncode!=0,index
print('parallel and invalid shard indexes passed')
