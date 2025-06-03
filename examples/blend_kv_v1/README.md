# Examples vLLM + LMCache w. CacheBlend
LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).
## CPU offloading
- `python blend.py` - CacheBlend with CPU as backend
## Disk offloading
- `python blend.py --use-disk` - CachBlend with local disk as backend