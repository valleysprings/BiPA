"""Native protocol freeze, resume, and rejection of partial outcome panels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.core.native_protocol import prepare, require_complete


def test_protocol_fresh_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.native_protocol.source_files", lambda root: [])
    out = tmp_path / 'run'
    args = SimpleNamespace(arm='seed', protocol='native-v1', feedback_mode='full',
                           model='offline', benchmark='tau3-bench', harness='opencode',
                           scale='smoke', cache_root=tmp_path / 'cache', resume=False)
    sample = {'train': ['a'], 'sel': ['b'], 'test': ['c']}
    saved = prepare(args, out, sample, [['a']], 'seed')
    args.resume = True
    assert prepare(args, out, sample, [['a']], 'different seed') == saved  # a resumed run keeps its recorded protocol


def test_complete_requires_every_task_row():
    require_complete([{'id': 'a', 'success': 0}, {'id': 'b', 'success': 1}], ['a', 'b'])
    require_complete([{'id': 'a', 'success': None}], ['a'])  # an unknown outcome is recorded, not a stop
    with pytest.raises(RuntimeError):
        require_complete([{'id': 'a', 'success': 1}], ['a', 'b'])
    with pytest.raises(RuntimeError):
        require_complete([{'id': 'a', 'success': 1}, {'id': 'a', 'success': 0}], ['a', 'b'])
    with pytest.raises(RuntimeError):
        require_complete([{'id': 'a', 'repetition': 1, 'success': 1}], ['a'])
