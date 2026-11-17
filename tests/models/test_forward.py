import pathlib, subprocess, tempfile, sys, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
exe, model_path = sys.argv[1:3]
atol = float(sys.argv[3]) if len(sys.argv)>3 else 0.0003
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
    attn_implementation='eager', local_files_only=True).cuda().eval()
with tempfile.TemporaryDirectory() as temp:
    def run(ids):
        path=pathlib.Path(temp)/'logits.bin'
        subprocess.run([exe,'--max-context','256','--model',model_path,'--tokens',','.join(map(str,ids)),
                        '--output',str(path)], check=True)
        actual=np.fromfile(path,dtype=np.float32).reshape(len(ids),-1)
        with torch.no_grad():
            expected=model(torch.tensor([ids],device='cuda'),use_cache=False).logits[0].float().cpu().numpy()
        assert actual.shape==expected.shape
        assert np.isfinite(actual).all()
        error=np.abs(actual-expected)
        print('tokens',len(ids),'max_abs',float(error.max()),'mean_abs',float(error.mean()),flush=True)
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=0.0001)
        assert actual[-1].argmax()==expected[-1].argmax()
        return actual
    a=run([9707])
    b=run([9707,11,1879,0])
    run([151644,872,198,9707,11,1879,0,151645])
    run(([9707,11,1879,0]*8))
    run(([9707,11,1879,0]*64))
    np.testing.assert_allclose(a[0],b[0],atol=atol,rtol=0.0001)
    for tokens in ['-1','151936','abc','1,','',','.join(['1']*257)]:
        r=subprocess.run([exe,'--max-context','256','--model',model_path,'--tokens',tokens],capture_output=True)
        assert r.returncode!=0,tokens
    bad=pathlib.Path(temp)/'bad-model'
    bad.mkdir()
    (bad/'model.safetensors').symlink_to(pathlib.Path(model_path).resolve()/'model.safetensors')
    original=json.loads((pathlib.Path(model_path)/'config.json').read_text())
    for key,value in [('model_type','llama'),('hidden_size',32),('attention_bias',True),
                      ('rope_scaling',{'rope_type':'linear','factor':2}),('head_dim',127)]:
        config=dict(original); config[key]=value
        (bad/'config.json').write_text(json.dumps(config))
        r=subprocess.run([exe,'--max-context','256','--model',str(bad),'--tokens','1'],capture_output=True)
        assert r.returncode!=0,(key,r.stdout)
print('Forward reference and input/config validation passed')
