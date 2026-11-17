import pathlib, subprocess, tempfile, numpy as np
exe='build/bin/qwecpp_forward'; model='/home/hzy/models/Qwen/Qwen3-0.6B'
with tempfile.TemporaryDirectory() as d:
 def run(n,args):
  out=d+'/out'
  r=subprocess.run([exe,'--model',model,'--tokens',','.join(map(str,([9707,11,1879,0]*64)[:n])),'--output',out]+args,capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  return np.fromfile(out,dtype=np.float32)
 for n in [1,4,32,256]:
  ref=run(n,['--devices','0'])
  for devices in ['0,1','1,0']:
   actual=run(n,['--devices',devices,'--parallel','tp'])
   np.testing.assert_allclose(actual,ref,atol=3e-4,rtol=1e-4)
   assert actual.reshape(n,-1)[-1].argmax()==ref.reshape(n,-1)[-1].argmax()
   print(n,devices,float(np.max(np.abs(actual-ref))),flush=True)
 for args in [['--devices','0'],['--devices','0,1','--split-layer','14']]:
  r=subprocess.run([exe,'--model',model,'--tokens','1','--parallel','tp']+args,capture_output=True)
  assert r.returncode!=0
