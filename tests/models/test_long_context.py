import subprocess,tempfile,pathlib,numpy as np
exe='build/bin/qwecpp_forward';model='/home/hzy/models/Qwen/Qwen3-0.6B'
with tempfile.TemporaryDirectory() as d:
 args=[exe,'--model',model,'--tokens',','.join(['9707']*257),'--max-context','512','--output',d+'/out']
 p=subprocess.run(args,capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 a=np.fromfile(d+'/out',dtype=np.float32).reshape(257,-1)
 assert np.isfinite(a).all()
 p=subprocess.run([exe,'--model',model,'--tokens','9707','--output',d+'/first'],capture_output=True)
 assert p.returncode==0
 np.testing.assert_allclose(a[0],np.fromfile(d+'/first',dtype=np.float32),atol=3e-4,rtol=1e-4)
 for limit in ['0','8193','abc']:
  p=subprocess.run([exe,'--model',model,'--tokens','1','--max-context',limit],capture_output=True)
  assert p.returncode!=0
 p=subprocess.run([exe,'--model',model,'--tokens',','.join(['1']*513),'--max-context','512'],capture_output=True)
 assert p.returncode!=0
print('long context and limits passed')
