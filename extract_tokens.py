import re
from pathlib import Path
b=Path('splib7.sgds').read_bytes()
# find ASCII words of length>=4
words=set()
for m in re.finditer(rb'[A-Za-z0-9_\-]{4,}', b):
    words.add(m.group(0).decode('ascii', errors='ignore'))
# show some likely mineral names by filtering for lowercase or known tokens
candidates=[w for w in sorted(words) if len(w)>4]
print('total tokens',len(candidates))
for w in candidates[:400]:
    print(w)
