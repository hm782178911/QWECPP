import pathlib,tempfile,subprocess,json
import numpy as np
from PIL import Image
import torch
from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
import transformers.models.qwen3_5.modeling_qwen3_5 as impl
model_dir='/home/hzy/models/Qwen/Qwen3.5-0.8B'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval()
for layer in model.model.language_model.layers:
 if hasattr(layer,'linear_attn'):
  layer.linear_attn.causal_conv1d_fn=None
  layer.linear_attn.chunk_gated_delta_rule=impl.torch_chunk_gated_delta_rule
  norm=impl.Qwen3_5RMSNormGated(128,eps=1e-6).cuda();norm.weight.data.copy_(layer.linear_attn.norm.weight.data);layer.linear_attn.norm=norm
with tempfile.TemporaryDirectory() as tmp:
 d=pathlib.Path(tmp)
 for height,width,fmt in [(256,256,'png'),(256,384,'png'),(73,97,'png'),(640,800,'png'),(256,256,'jpeg')]:
  y,x=np.indices((height,width));rgb=np.stack([(x*7)%256,(y*9)%256,(x*3+y*5)%256],axis=-1).astype('uint8')
  if fmt=='jpeg':rgb[:]=[64,128,192]
  image_path=d/('image.'+fmt);Image.fromarray(rgb).save(image_path)
  p=subprocess.run(['build/bin/qwecpp_image','--model',model_dir,'--image',str(image_path),'--prompt','描述这张图片。','--patches',str(d/'patches.bin'),'--features',str(d/'features.bin'),'--output',str(d/'logits.bin'),'--metadata',str(d/'metadata.json')],capture_output=True,text=True)
  assert p.returncode==0,p.stderr
  meta=json.loads((d/'metadata.json').read_text());image=Image.open(image_path).convert('RGB')
  inputs=processor.image_processor(images=[image],size={'shortest_edge':65536,'longest_edge':262144},return_tensors='pt')
  pixels=inputs['pixel_values'].cuda();grid=inputs['image_grid_thw'].cuda()
  assert grid.tolist()==[[1,meta['grid_h'],meta['grid_w']]]
  actual=np.fromfile(d/'patches.bin',dtype=np.float32).reshape(-1,1536)
  np.testing.assert_allclose(actual,pixels.cpu().numpy(),atol=2e-7,rtol=1e-6)
  with torch.no_grad():features=model.model.visual(pixels,grid_thw=grid).pooler_output.cpu().numpy()
  native_features=np.fromfile(d/'features.bin',dtype=np.float32).reshape(-1,1024)
  np.testing.assert_allclose(native_features,features,atol=0.0003,rtol=0.0001)
  ids=torch.tensor([meta['tokens']],device='cuda');types=(ids==248056).int()
  with torch.no_grad():logits=model(input_ids=ids,pixel_values=pixels,image_grid_thw=grid,mm_token_type_ids=types,use_cache=False).logits[0,-1].cpu().numpy()
  actual=np.fromfile(d/'logits.bin',dtype=np.float32)
  print(height,width,fmt,'patches',len(pixels),'features_max',float(np.abs(features-native_features).max()),'logits_max',float(np.abs(logits-actual).max()),'argmax',int(actual.argmax()),int(logits.argmax()),flush=True)
  np.testing.assert_allclose(actual,logits,atol=0.0003,rtol=0.0001);assert actual.argmax()==logits.argmax()
 # Two images require independent vision attention segments and correct multi-image MRoPE.
 first=Image.new('RGB',(256,256),(255,0,0));second=Image.new('RGB',(256,384),(0,0,255));first.save(d/'first.png');second.save(d/'second.png')
 p=subprocess.run(['build/bin/qwecpp_image','--model',model_dir,'--image',str(d/'first.png'),'--image2',str(d/'second.png'),'--prompt','对比两张图片。','--features',str(d/'features.bin'),'--output',str(d/'logits.bin'),'--metadata',str(d/'metadata.json')],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 meta=json.loads((d/'metadata.json').read_text());inputs=processor.image_processor(images=[first,second],return_tensors='pt');pixels=inputs['pixel_values'].cuda();grid=inputs['image_grid_thw'].cuda();assert meta['grids']==grid.tolist()
 ids=torch.tensor([meta['tokens']],device='cuda')
 with torch.no_grad():
  features=model.model.visual(pixels,grid_thw=grid).pooler_output.cpu().numpy()
  logits=model(input_ids=ids,pixel_values=pixels,image_grid_thw=grid,mm_token_type_ids=(ids==248056).int(),use_cache=False).logits[0,-1].cpu().numpy()
 actual=np.fromfile(d/'logits.bin',dtype=np.float32);actual_features=np.fromfile(d/'features.bin',dtype=np.float32).reshape(-1,1024)
 np.testing.assert_allclose(actual_features,features,atol=0.0003,rtol=0.0001);np.testing.assert_allclose(actual,logits,atol=0.0003,rtol=0.0001);assert actual.argmax()==logits.argmax()
 print('two images logits_max',float(np.abs(actual-logits).max()),flush=True)
 bad=d/'bad.png';bad.write_bytes(b'not an image')
 p=subprocess.run(['build/bin/qwecpp_image','--model',model_dir,'--image',str(bad)],capture_output=True)
 assert p.returncode!=0
print('native image preprocessing, vision features, MRoPE and multimodal logits reference passed')
