import numpy as np
import h5py

no_traces=10000
rounds=22
trace_length=rounds*4
mask=(1 << 8)-1
mask2=(1 << 16)-1
out_file="speck32_traces_88_10k.h5"

fixed_key=0x10


def rol8(x, r):
    return ((x<<r)|(x>>(8-r)))&mask

def ror8(x, r):
    return ((x>>r)|(x<<(8-r)))&mask

def hw(x):

    return bin(x&0xFF).count("1")


def key_schedule(k):
    return [(k&0xFF) for _ in range(rounds)]


def encrypt_and_trace(pt, rks):
    x=(pt>>16)&mask2
    y=pt&mask2

    xh, xl=(x>>8)&mask, x&mask
    yh, yl=(y>>8)&mask, y&mask

    trace=[]
    for rk in rks:
        xr=ror8(x, 7)
        trace.append(hw(xr))
        xa=(xr + y) & mask
        trace.append(hw(xa))
        xh=xa ^ rk
        trace.append(hw(xh))
        yl=rol8(yl, 2) ^ xh
        trace.append(hw(yl))
    ct=((xh<<24)|(xl<<16)|(yh<<8)|yl)
    return ct, trace

ks=key_schedule(fixed_key)
pts=np.random.randint(0, 2**32, size=no_traces, dtype=np.uint32)
cts=np.zeros(no_traces, dtype=np.uint32)
traces=np.zeros((no_traces, trace_length), dtype=np.uint8)

for i in range(no_traces):
    c, t=encrypt_and_trace(int(pts[i]), ks)
    cts[i]=c
    traces[i]=t

with h5py.File(out_file, "w") as f:
    f.create_dataset("plaintexts", data=pts)
    f.create_dataset("ciphertexts", data=cts)
    f.create_dataset("traces", data=traces)
    f.create_dataset("key", data=np.array([fixed_key], dtype=np.uint16))

print(" Dataset saved as", out_file)