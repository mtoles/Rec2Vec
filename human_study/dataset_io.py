"""Publish small human datasets while retaining prior versions and stable payloads."""

import json
from datetime import datetime, timezone

from datasets import load_from_disk


def save_version(dataset, path, report=None, report_name='combination_report.json'):
    unchanged = False
    if path.exists():
        previous = load_from_disk(str(path))
        unchanged = previous.features == dataset.features and previous.to_dict() == dataset.to_dict()
        del previous
        if not unchanged:
            archive = path.parent / 'old' / (path.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            archive.parent.mkdir(parents=True, exist_ok=True)
            path.rename(archive)
            print(f'Archived {archive}', flush=True)
    if not unchanged:
        dataset.save_to_disk(str(path))
    if report is not None:
        (path / report_name).write_text(json.dumps(report, indent=2) + '\n')
    return not unchanged
