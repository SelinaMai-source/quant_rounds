"""Validate actual transitive NN input bytes before constructing any cache key."""
import json
from pathlib import Path
from core.research_integrity_v5.cache import identity

def _validate(directory,names=None):
    directory=Path(directory)
    marker=json.loads((directory/'COMPLETE.json').read_text())
    expected={item['path']:item for item in marker['artifacts']}
    selected=sorted(expected) if names is None else names
    records=[identity(directory/'COMPLETE.json')]
    for name in selected:
        assert name in expected, 'Unregistered required NN input: '+name
        actual=identity(directory/name);record=expected[name]
        assert actual['sha256']==record['sha256'] and actual['bytes']==record['bytes'], 'Changed completed NN input; reject stale cache before reuse: '+str(directory/name)
        records.append(actual)
    return records

def validate_store(path):
    path=Path(path)
    records=_validate(path)
    meta=json.loads((path/'STORE.json').read_text())
    records+=_validate(meta['graphs'])
    records+=_validate(meta['provider'],['calendars/day.txt','instruments/all.txt'])
    records+=_validate(meta['market'])
    return records
