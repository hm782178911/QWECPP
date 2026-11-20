import json, pathlib, struct, subprocess, sys, tempfile
exe = sys.argv[1]
cases = 0
with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    def run(header, data=b'', ok=True, raw=False):
        global cases
        cases += 1
        h = header if raw else json.dumps(header, separators=(',', ':')).encode()
        f = root / 'model.safetensors'
        f.write_bytes(struct.pack('<Q', len(h)) + h + data)
        r = subprocess.run([exe, '--model', str(root)], capture_output=True, text=True)
        assert (r.returncode == 0) == ok, (header, r.stdout, r.stderr)
        return r.stdout
    def entry(dtype='F32', shape=None, offsets=None):
        return dict(dtype=dtype, shape=[2] if shape is None else shape,
                    data_offsets=[0,8] if offsets is None else offsets)
    out = run({'x': entry(), '__metadata__': {'format':'pt'}}, struct.pack('<ff',1,2))
    assert 'Tensor Count: 1' in out and 'Total Parameters: 2' in out
    assert 'Total Weight Size: 8' in out and 'FP32' in out
    subprocess.run([sys.argv[2], str(root/'model.safetensors')], check=True)
    run({'scalar': entry(shape=[], offsets=[0,4])}, b'1234')
    run({'empty': entry(shape=[0], offsets=[0,0])})
    run({'rank5': entry(shape=[1]*5, offsets=[0,4])}, bytes(4))
    run({'rank8': entry(shape=[1]*8, offsets=[0,4])}, bytes(4))
    run({'rank9': entry(shape=[1]*9, offsets=[0,4])}, bytes(4), False)
    run({})
    for dtype, size in [('F32',4),('F16',2),('BF16',2),('I8',1),('I32',4),('I64',8)]:
        run({'x':entry(dtype,[1],[0,size])}, bytes(size))
    run({'x':entry(), 'y':entry()}, bytes(8), False)
    for e in [entry('BAD'), entry(shape=[-1]), entry(shape=[1.5]),
              entry(shape=[1,1,1,1,1]), entry(offsets=[8,0]),
              entry(offsets=[0,9]), entry(offsets=[1,9]),
              entry(shape=[9007199254740992]), entry(shape=[4294967296,4294967296]),
              entry(offsets=[0,7]), {'dtype':'F32','shape':[2]}]:
        run({'x':e}, bytes(8), False)
    run({'x':entry()}, bytes(9), False)
    run({'__metadata__':{'x':1}}, ok=False)
    for raw in [b'[]', b'{', b'{}junk', b'{"x":{"dtype":"F32","dtype":"F32","shape":[2],"data_offsets":[0,8]}}',
                b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}']:
        run(raw, bytes(8), False, True)
    f = root/'model.safetensors'
    for raw in [b'', b'123', struct.pack('<Q', 100000001), struct.pack('<Q',999)+b'{}']:
        f.write_bytes(raw)
        assert subprocess.run([exe,'--model',str(f)],capture_output=True).returncode != 0
    assert subprocess.run([exe,'--model',str(root/'missing')],capture_output=True).returncode != 0
print('Safetensors integration cases passed:', cases)
