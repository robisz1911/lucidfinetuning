import sys

ls = []
for l in sys.stdin:
    dir, layer, size = l.strip().split()
    size = int(size)
    ls.append({ "layer": dir + "/" + layer, "size": str(size) })

import json

print("layers =")

print(json.dumps(ls, sort_keys=True, indent=4))
