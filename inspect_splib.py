from pathlib import Path
p=Path('splib7.sgds')
print('exists',p.exists())
with p.open('rb') as f:
    head=f.read(200)
print('len head',len(head))
print(head[:200])
# try small heuristics
s = head[:200]
try:
    import gzip
    if s[:2]==b'\x1f\x8b':
        print('gzip header detected')
except Exception:
    pass
try:
    import pickle
    with p.open('rb') as f:
        obj = pickle.load(f)
    print('pickle loaded, type', type(obj))
except Exception as e:
    print('pickle failed',e)
