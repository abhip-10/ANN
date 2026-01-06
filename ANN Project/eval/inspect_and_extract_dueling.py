import torch
import pprint
import numpy as np
from pathlib import Path

ckpt_path = Path(r"C:\ANN\ANN Project\dueling_double_dqn_highway.pth")
out_path = Path(r"C:\ANN\ANN Project\models\dueling_policy_extracted.pth")

print('Checkpoint:', ckpt_path)

# Add safe global required by the checkpoint
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    print('Added safe global for numpy._core.multiarray.scalar')
except Exception as e:
    print('Failed to add safe global:', e)

try:
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
except Exception as e:
    print('Weights-only load failed:', e)
    # Try unsafe load only if weights-only failed
    try:
        print('Trying full load (unsafe). Only proceed if you trust the file)')
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    except Exception as e2:
        print('Full load failed too:', e2)
        raise SystemExit(1)

print('Loaded object type:', type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print('Top-level keys:')
    pprint.pprint(keys)

    # candidate keys to extract
    candidates = ['policy_net', 'policy', 'policy_state_dict', 'model_state_dict', 'state_dict', 'q_net', 'net', 'policy_state']
    found = False
    for k in candidates:
        if k in ckpt and isinstance(ckpt[k], dict):
            torch.save(ckpt[k], str(out_path))
            print(f'Extracted "{k}" to {out_path}')
            found = True
            break
    if not found:
        # try to find any dict of tensors in nested keys
        for k,v in ckpt.items():
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                torch.save(v, str(out_path))
                print(f'Extracted nested dict at "{k}" to {out_path}')
                found = True
                break
    if not found:
        print('No suitable policy dict found to extract.')
else:
    print('Checkpoint is not a dict; nothing to extract')
