import pathlib,subprocess,sys,tempfile
import numpy as np
exe,model=sys.argv[1:3]
with tempfile.TemporaryDirectory() as tmp:
 def run(ids,extra):
  output=pathlib.Path(tmp)/'logits.bin'
  r=subprocess.run([exe,'--model',model,'--tokens',','.join(map(str,ids)),'--output',str(output)]+extra,capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  return np.fromfile(output,dtype=np.float32)
 for length in [1,4,256]:
  ids=([9707,11,1879,0]*64)[:length]
  single=run(ids,['--devices','0'])
  for extra in [['--devices','0,1'],['--devices','1,0'],['--devices','0,1','--pp-host-transfer']]:
   actual=run(ids,extra)
   np.testing.assert_allclose(actual,single,atol=3e-4,rtol=1e-4)
   print('PP comparison:',length,extra,'max_abs',float(np.max(np.abs(actual-single))),flush=True)
 ids=[9707,11,1879,0];single=run(ids,['--devices','0'])
 for split in [1,27]:
  np.testing.assert_allclose(run(ids,['--devices','0,1','--split-layer',str(split)]),single,atol=3e-4,rtol=1e-4)
 for extra in [['--devices','0,0'],['--devices','999'],['--devices','0,1','--split-layer','0'],['--devices','0,1','--split-layer','28'],['--devices','0','--split-layer','14']]:
  r=subprocess.run([exe,'--model',model,'--tokens','1']+extra,capture_output=True)
  assert r.returncode!=0,extra
print('PP correctness, transfer fallback, stage boundaries and invalid options passed')
