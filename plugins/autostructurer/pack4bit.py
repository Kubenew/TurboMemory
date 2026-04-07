import numpy as np

def pack_4bit(vec: np.ndarray):
    v=vec.astype(np.float32)
    vmin=float(v.min()); vmax=float(v.max())
    if vmax-vmin < 1e-9:
        scale=1.0; zero=vmin
        q=np.zeros_like(v, dtype=np.uint8)
    else:
        scale=(vmax-vmin)/15.0
        zero=vmin
        q=np.clip(np.round((v-zero)/scale),0,15).astype(np.uint8)
    if len(q)%2==1:
        q=np.append(q,0)
    packed=(q[0::2]&0x0F)|((q[1::2]&0x0F)<<4)
    return packed.tobytes(), scale, zero

def unpack_4bit(packed_bytes: bytes, dim: int, scale: float, zero: float):
    packed=np.frombuffer(packed_bytes, dtype=np.uint8)
    q0=packed&0x0F
    q1=(packed>>4)&0x0F
    q=np.empty(len(packed)*2, dtype=np.uint8)
    q[0::2]=q0; q[1::2]=q1
    q=q[:dim]
    return (q.astype(np.float32)*scale+zero).astype(np.float32)