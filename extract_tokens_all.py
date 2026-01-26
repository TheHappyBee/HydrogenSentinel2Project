from pathlib import Path
import re
for fn in ['splib7.sgdr','splib7.sgdd']:
    p=Path(fn)
    if not p.exists():
        print(fn,'missing')
        continue
    b=p.read_bytes()
    words=set()
    for m in re.finditer(rb'[A-Za-z0-9_\-]{4,}', b):
        words.add(m.group(0).decode('ascii', errors='ignore'))
    cand=sorted(words)
    print(fn,'tokens',len(cand))
    print('\n'.join(cand[:200]))
