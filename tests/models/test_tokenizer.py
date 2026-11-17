import json, subprocess, sys
from transformers import AutoTokenizer
exe, path=sys.argv[1:3]
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
samples=['','Hello, world!','你好，世界！','你好\n世界\t 123456','I\'m testing isn\'t it?','😀🚀 café e\u0301','  \n\n  x\r\ny',
         '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
         '<tool_call>{"x":1}</tool_call>','中文English混合123 ！', 'a'*200]
for text in samples:
    result=subprocess.run([exe,path,text],check=True,capture_output=True,text=True)
    actual=json.loads(result.stdout)
    expected=tok.encode(text,add_special_tokens=False)
    assert actual['ids']==expected,(text,actual['ids'],expected)
    assert actual['decoded']==tok.decode(expected,skip_special_tokens=False),(text,actual)
print('Tokenizer reference cases passed:',len(samples))
