import subprocess,time,urllib.request,urllib.error,json,socket,base64,io
from PIL import Image
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
p=subprocess.Popen(['build/bin/qwecpp_server','--model','/home/hzy/models/Qwen/Qwen3.5-0.8B','--port',str(port),'--web-root','web'],stdout=subprocess.DEVNULL)
try:
 for _ in range(100):
  try:caps=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/capabilities',timeout=1));break
  except Exception:time.sleep(.2)
 else:raise AssertionError('startup timeout')
 assert caps['vision'] and not caps['audio'] and caps['max_images']==4
 data=io.BytesIO();Image.new('RGB',(256,256),(255,0,0)).save(data,format='PNG')
 uri='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
 def image(url):return {'type':'image_url','image_url':{'url':url}}
 def request(parts):
  body={'id':'vision','messages':[{'role':'user','content':parts}],'max_new_tokens':2}
  return urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=90)
 response=request([image(uri),{'type':'text','text':'图片主要是什么颜色？'}]).read().decode()
 events=[json.loads(s[6:]) for s in response.splitlines() if s.startswith('data: ')]
 assert events[-1]['type']=='done',events
 m=events[-1]['metrics'];assert m['image_count']==1 and m['image_tokens']==64 and m['vision_ms']>0 and m['output_tokens']>0,m
 response=request([image(uri),image(uri),{'type':'text','text':'对比这两张图。'}]).read().decode()
 events=[json.loads(s[6:]) for s in response.splitlines() if s.startswith('data: ')]
 assert events[-1]['type']=='done' and events[-1]['metrics']['image_count']==2 and events[-1]['metrics']['image_tokens']==128,events
 for parts in [[image('https://example.invalid/a.png')],[image('data:image/png;base64,AAAA')],[image('data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode())],[image(uri)]*5,[{'type':'video_url','video_url':{'url':uri}}]]:
  try:request(parts);raise AssertionError('invalid image input accepted')
  except urllib.error.HTTPError as e:assert e.code==400
 # Bad request leaves model available for a following ordinary text request.
 response=request([{'type':'text','text':'你好'}]).read().decode();assert '"type":"done"' in response
 print('image capabilities, one/two-image SSE, metrics, URL/MIME/malformed/limit rejection and recovery passed')
finally:p.terminate();p.wait(timeout=10)
