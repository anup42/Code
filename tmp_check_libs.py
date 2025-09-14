import importlib, sys
mods = ['yaml','numpy']
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('MISS', m, repr(e))
