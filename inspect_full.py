import h5py, sys

path = sys.argv[1]
f = h5py.File(path, "r")
keys = list(f.keys())
print(f"Top-level: {len(keys)} keys; first few: {keys[:5]}")
print("Top-level attrs:", {k: f.attrs[k] for k in f.attrs})

print("---- First Few Branches ----")

def peek(g, prefix=" ", depth=0, max_depth=3):
    for k in list(g.keys()):
        obj = g[k]
        if isinstance(obj, h5py.Dataset):
            print(f"{prefix}{k}: shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{prefix}{k}/ (group, {len(obj.keys())} children)")
            if depth < max_depth:
                peek(obj, prefix + " ", depth + 1, max_depth)

peek(f)
f.close()