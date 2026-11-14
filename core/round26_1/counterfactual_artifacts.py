"""Export completed diagnostic evidence; never changes or trains a model."""
from pathlib import Path
import gzip
import json
import os
import shutil
from core.round26_1.market import R
from core.research_integrity_v5.cache import identity, sha


def export_completed():
    exported = []
    for result_path in sorted((R / 'generation_counterfactual_tests').glob('*/*/RESULT.json')):
        result = json.loads(result_path.read_text())
        if result['status'] != 'PASS_FULL_ORIGINAL_PPO_UNAVAILABLE_TARGET_COUNTERFACTUAL':
            continue
        source = Path(result['diagnostic'])
        complete = json.loads((source / 'COMPLETE.json').read_text())
        for name, key in [('CONTRACT.json', 'contract_sha256'),
                          ('FROZEN_FACTOR.json', 'factor_model_sha256'),
                          ('factor_predictions.parquet', 'factor_prediction_sha256')]:
            assert sha(source / name) == complete[key], (source, name)
        destination = result_path.parent / 'completed_artifacts'
        destination.mkdir(parents=True, exist_ok=True)
        records = []
        for name in ['CONTRACT.json', 'FROZEN_FACTOR.json', 'SELECTED_CHECKPOINT.json',
                     'factor_predictions.parquet', 'RESOURCES.json', 'RESOLVED_PPO.json',
                     'COMPLETE.json']:
            path = source / name
            assert path.is_file(), path
            target = destination / name
            temporary = target.with_suffix(target.suffix + '.partial')
            shutil.copyfile(path, temporary)
            os.replace(temporary, target)
            assert sha(target) == sha(path)
            records.append({'source': identity(path), 'uploaded': identity(target)})
        attempts = source / 'all_expression_evaluations.jsonl'
        target = destination / 'all_expression_evaluations.jsonl.gz'
        with attempts.open('rb') as src, target.with_suffix('.partial').open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as out:
                shutil.copyfileobj(src, out)
        os.replace(target.with_suffix('.partial'), target)
        records.append({'source': identity(attempts), 'uploaded': identity(target),
                        'encoding': 'gzip; all source attempts, including rejection/error'})
        large = []
        for path in sorted((source / 'resume_slots').rglob('*')):
            if path.is_file():
                large.append({'schema': 'original SB3 policy zip / own AST+RNG resume state',
                              **identity(path)})
        assert large, 'Completed full PPO test must retain its actual checkpoints'
        manifest = {'scope': 'Completed diagnostic model only; no new economic trial',
                    'result': identity(result_path), 'uploaded_artifacts': records,
                    'large_server_artifacts': large}
        path = destination / 'manifest.json'
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
        exported.append(str(path))
    return exported


if __name__ == '__main__':
    print(json.dumps(export_completed(), ensure_ascii=False))
