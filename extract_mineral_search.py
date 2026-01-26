from pathlib import Path
files=['splib7.sgds','splib7.sgdr','splib7.sgdd']
minerals=['hematite','goethite','magnetite','olivine','pyroxene','kaolinite','illite','montmorillonite','chlorite','limonite','biotite','muscovite','clinochlore','serpentine','jarosite']
for fn in files:
    p=Path(fn)
    if not p.exists():
        continue
    b=p.read_bytes().lower()
    print('\nSearching',fn)
    for m in minerals:
        if m.encode('ascii') in b:
            print('  found',m)
