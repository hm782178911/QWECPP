import subprocess,time,urllib.request,json,socket,sys
exe='build/bin/qwecpp_server'
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
p=subprocess.Popen([exe,'--model',(sys.argv[1] if len(sys.argv)>1 else '/home/hzy/models/Qwen/Qwen3.5-0.8B'),'--port',str(port),'--web-root','web']+sys.argv[2:],stdout=subprocess.DEVNULL)
try:
 for _ in range(100):
  try:
   caps=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/capabilities',timeout=1));break
  except Exception:time.sleep(.2)
 else:raise AssertionError('server startup timeout')
 assert isinstance(caps['vision'],bool) and caps['stream']==True
 def post(body):
  return urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=60).read().decode()
 text=post({'id':'test','messages':[{'role':'user','content':'你好'}],'max_new_tokens':4})
 events=[json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
 assert events[-1]['type']=='done' and events[-1]['metrics']['output_tokens']>0,events
 for body in [{'messages':[]},{'id':'bad','messages':[{'role':'user','content':[{'type':'image_url'}]}]}]:
  try:post(body);raise AssertionError('invalid request accepted')
  except urllib.error.HTTPError as e:assert e.code==400
 assert b'QWECPP' in urllib.request.urlopen(f'http://127.0.0.1:{port}/').read()
 # Hold a real stream open, reject concurrent generation, cancel, then recover.
 req=urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=json.dumps({'id':'cancel-me','messages':[{'role':'user','content':'详细介绍大语言模型，请尽量详细。'}],'max_new_tokens':256}).encode(),headers={'Content-Type':'application/json'})
 stream=urllib.request.urlopen(req,timeout=60)
 first=stream.readline().decode();assert 'delta' in first,first
 try:post({'id':'busy','messages':[{'role':'user','content':'你好'}]});raise AssertionError('busy accepted')
 except urllib.error.HTTPError as e:assert e.code==409
 urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/cancel',data=b'{"id":"cancel-me"}',headers={'Content-Type':'application/json'})).read()
 rest=stream.read().decode();assert 'cancelled' in rest,rest
 stream.close()
 time.sleep(.1)
 text=post({'id':'after-cancel','messages':[{'role':'user','content':'你好'}],'max_new_tokens':1})
 assert 'done' in text
 # A foreign browser origin cannot submit to this local service.
 bad=urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://example.invalid'})
 try:urllib.request.urlopen(bad);raise AssertionError('foreign origin accepted')
 except urllib.error.HTTPError as e:assert e.code==403

 print('capabilities, stream, metrics, invalid requests, concurrency, cancellation, recovery and origin checks passed')
finally:p.terminate();p.wait(timeout=10)
