"""Restricted, portable state-dictionary checkpoints."""
import json
from pathlib import Path
import torch
from .model import AODNet

def save(path, model, metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':1,'architecture':'AODNet','state_dict':model.state_dict(),'metadata':json.loads(json.dumps(metadata))},path)

def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('format_version')!=1 or payload.get('architecture')!='AODNet':
        raise ValueError('Expected a project AODNet state-dictionary checkpoint')
    model=AODNet();model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    return model,payload.get('metadata',{})
