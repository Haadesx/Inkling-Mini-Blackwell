#!/usr/bin/env python3
import json, struct, sys
import numpy as np

def load(path):
    with open(path,'rb') as f:
        h=f.read(32)
        if len(h)!=32: raise ValueError('missing header')
        magic,v,n,chunk,gpu,offset,decode,threads=struct.unpack('=8i',h)
        if magic!=0x494c4731 or not 0<v<10000000 or not 0<n<10000000: raise ValueError('invalid header')
        tokens=np.fromfile(f,dtype=np.int32,count=n)
        if len(tokens)!=n: raise ValueError('missing tokens')
        data=np.fromfile(f,dtype=np.float32)
        if len(data)!=n*v: raise ValueError('missing or extra logits')
        data=data.reshape(n,v)
        if np.isnan(data).any() or np.isposinf(data).any(): raise ValueError('NaN or positive infinity')
        return (v,n,chunk,gpu,offset,decode,threads),tokens,data

a,b=map(load,sys.argv[1:]); meta,t,x=a; meta2,t2,y=b
if meta!=meta2 or not np.array_equal(t,t2): raise ValueError('metadata or tokens differ')
if not np.array_equal(np.isneginf(x),np.isneginf(y)): raise ValueError('negative-infinity mask differs')
rows=[]
for i,(u,v) in enumerate(zip(x,y)):
    mask=np.isfinite(u); p=u[mask].astype(np.float64); q=v[mask].astype(np.float64)
    if not len(p): raise ValueError('all logits masked')
    d=p-q; ep=np.exp(p-p.max()); eq=np.exp(q-q.max()); pp=ep/ep.sum(); qq=eq/eq.sum()
    kl=float(np.sum(pp*((p-p.max())-np.log(ep.sum())-(q-q.max())+np.log(eq.sum()))))
    rows.append(dict(row=i, position=i+meta[4], token=int(t[i]), argmax_equal=int(np.argmax(u))==int(np.argmax(v)), kl=kl, max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d*d))), variance_a=float(np.var(p)), variance_b=float(np.var(q)), bitwise_equal=bool(np.array_equal(u.view(np.uint32),v.view(np.uint32))), masked=int((~mask).sum())))
print(json.dumps(dict(metadata=dict(zip(['vocab','tokens','chunk','gpu_layers','offset','decode_tokens','threads'],meta)), token_ids=t.tolist(), all_argmax_equal=all(r['argmax_equal'] for r in rows), all_bitwise_equal=all(r['bitwise_equal'] for r in rows), negative_infinity_mask_identical=True, nan_or_positive_infinity=False, max_abs=max(r['max_abs'] for r in rows), max_rms=max(r['rms'] for r in rows), max_kl=max(r['kl'] for r in rows), rows=rows),indent=2,allow_nan=False))
