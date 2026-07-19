# convert/ — HF → GGUF (offline only)

**Policy:** inspect → dump → convert → load-verify → parity.  
See [DEVELOPMENT.md](../DEVELOPMENT.md).

| Script | ARCH | Status |
|--------|------|--------|
| `inspect_safetensors.py` | — | Ready |
| `convert_t5_encoder.py` | `uniflow_t5enc` | Ready |
| `convert_dit.py` | `uniflow_dit` | Ready (T2A subset) |
| `convert_vae.py` | `uniflow_vae` | Ready (weight_norm fused) |
| `convert_instructions.py` | `uniflow_instructions` | Ready |
| `quantize.py` | — | Ready (generic) |

```bash
pip install -r convert/requirements.txt
# after downloading models/uniflow-small/
make convert-small
make test
```
