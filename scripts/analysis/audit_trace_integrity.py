#!/usr/bin/env python3
"""Read-only, offline audit of selected DeepSeek test episodes and original evidence."""
import argparse, collections, datetime, hashlib, json, math, importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bipa.layout import (BASELINE_ROOT, BASELINES_ROOT, MITIGATION_ROOT as ROOT,
                    ARTIFACTS, RESULTS, EXECUTION, ATTRIBUTION, PROFILING, CACHE, HARNESS_BENCH, relocate)

import math
def version(text):
    files = {} if text is None else {'SKILL.md': text}
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def episode_metrics(episode):
    """Exact graph total semantics: unique requests, executed calls, complete capture.

    Exposure allocation is unnecessary for N/X totals. Match graph.py by ignoring
    proposed/unexecuted calls and withholding N when execution status is unknown.
    """
    capture = episode.get('capture') or {}
    requests, seen = [], set()
    for req in episode.get('requests', []):
        identity = json.dumps(req.get('request_id'), sort_keys=True)
        if identity not in seen:
            seen.add(identity)
            requests.append(req)
    calls = [call for req in requests for call in req.get('tool_calls', [])]
    n = sum(call.get('executed') is True for call in calls)
    if capture.get('invocations_complete') is not True or any(
            call.get('executed') is not True and call.get('executed') is not False for call in calls):
        n = None
    inputs = [req.get('input_tokens') for req in requests]
    x = sum(inputs) if capture.get('requests_complete') is True and all(map(finite, inputs)) else None
    cost = n + x / 1000 if n is not None and x is not None else None
    return {'N': n, 'X': x, 'cost': cost, 'measurement': 'exact captured request totals; graph total semantics',
            'retry_overhead_excluded': bool(capture.get('retry_overhead'))}


def recovery_overlay(data, path):
    return {}

_spec = importlib.util.spec_from_file_location('native_trajectory', str(HARNESS_BENCH / 'src/model/trajectory.py'))
_trajectory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trajectory)

def load(path):
    p = relocate(path)
    if p.suffix == '.jsonl':
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())

class Evidence:
    def __init__(self): self.cache = {}
    def get(self, ref):
        path, _, fragment = str(ref).partition('#')
        if not fragment:
            for ext in ('.json/', '.jsonl/'):
                if ext in path:
                    head, tail = path.split(ext, 1); path = head + ext[:-1]; fragment = tail; break
        if path not in self.cache:
            try: self.cache[path] = (load(path) if Path(path).suffix in ('.json', '.jsonl') else relocate(path).read_text(), None)
            except (OSError, ValueError) as exc: self.cache[path] = (None, str(exc))
        value, error = self.cache[path]
        if error: raise ValueError(error)
        if fragment.startswith('line='):
            line, _, tail = fragment[len('line='):].partition('/')
            index = int(line) - 1
            if index < 0: raise ValueError('line pointers are one-based')
            value = value[index] if isinstance(value, list) else json.loads(value.splitlines()[index])
            fragment = tail
        if fragment == 'normalized' or fragment.startswith('normalized/'):
            key = path + '#normalized'
            if key not in self.cache: self.cache[key] = (_trajectory.unpack(value), None)
            value = self.cache[key][0]
            fragment = '' if fragment == 'normalized' else fragment[len('normalized/'):]
        for part in fragment.strip('/').split('/') if fragment else []:
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value

def refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('record_path','source_ref','source_path','entry_path','scored_attempt','results') and isinstance(item,str) and item.startswith('/'):
                yield item
            elif key in ('evidence_refs','attempts') and isinstance(item,list):
                yield from (v for v in item if isinstance(v,str) and v.startswith('/'))
            else: yield from refs(item)
    elif isinstance(value,list):
        for item in value: yield from refs(item)

def audit_episode(ep, evidence, expected_files):
    issues = []; checked = set()
    def check(ref):
        if ref in checked:
            try: return evidence.get(ref)
            except (ValueError, KeyError, IndexError, TypeError): return None
        checked.add(ref)
        try: return evidence.get(ref)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            issues.append({'kind':'evidence_unresolved','path':ref,'reason':str(exc)}); return None
    for ref in refs(ep): check(ref)
    for paths in (ep.get('skill_delivery') or {}).get('files',{}).values():
        for ref in paths: check(ref)
    capture = ep.get('capture') or {}
    source_capture = {}
    scored = capture.get('scored_attempt')
    if scored:
        source = check(scored)
        if isinstance(source, dict):
            extra = (_trajectory.unpack(source).get('extra') or {})
            source_capture = {k: extra[k] for k in ('capture_status','capture_exception_type','capture_exception_message') if k in extra}
    for key in ('requests_complete','invocations_complete'):
        if capture.get(key) is not True: issues.append({'kind':key,'path':scored,'reason':source_capture.get('capture_exception_message','capture flag is not true; see invocation execution witnesses'), 'source_capture':source_capture})
    if ep.get('success') is None or ep.get('outcome_valid') is not True:
        issues.append({'kind':'outcome_unknown_or_invalid','reason':ep.get('outcome_reason')})
    delivery = ep.get('skill_delivery') or {}
    if delivery.get('verified') is not True: issues.append({'kind':'skill_delivery_unverified'})
    requests = ep.get('requests',[])
    ids = [r.get('request_id') for r in requests]
    if len(ids)!=len(set(ids)): issues.append({'kind':'duplicate_request_id'})
    for req in requests:
        path = req.get('record_path')
        record = check(path) if path else None
        if not path: issues.append({'kind':'record_path_missing','request_id':req.get('request_id')})
        if isinstance(record,dict):
            if record.get('request_id')!=req.get('request_id'): issues.append({'kind':'request_id_mismatch','path':path})
            for ref in refs(record): check(ref)
            if record.get('usage',{}).get('input_tokens')!=req.get('input_tokens'):
                issues.append({'kind':'record_input_tokens_mismatch','path':path})
    for name, text in expected_files.items():
        paths = delivery.get('files',{}).get(name,[])
        found = False
        for ref in paths:
            value = check(ref)
            if text in json.dumps(value,ensure_ascii=False): found=True; break
            # json.dumps escapes newlines; compare decoded leaf strings as well.
            def strings(v):
                if isinstance(v,str): yield v
                elif isinstance(v,dict):
                    for x in v.values(): yield from strings(x)
                elif isinstance(v,list):
                    for x in v: yield from strings(x)
            if any(text in x or text.replace('"', '\\"') in x for x in strings(value)): found=True; break
        if not found: issues.append({'kind':'exact_skill_text_not_found','file':name})
    metrics = episode_metrics(ep)
    nx_reasons=[]
    if metrics['N'] is None:
        nx_reasons.append('invocations_incomplete' if capture.get('invocations_complete') is not True else 'execution_status_unknown')
    if metrics['X'] is None:
        nx_reasons.append('requests_incomplete' if capture.get('requests_complete') is not True else 'input_tokens_unknown')
    unknown_calls = [{'request_id':r.get('request_id'),'record_path':r.get('record_path'),'call_id':c.get('call_id'),'tool':c.get('tool'),'evidence_refs':c.get('evidence_refs',[])} for r in requests for c in r.get('tool_calls',[]) if c.get('executed') not in (True,False)]
    return {'task_id':ep['task_id'],'repetition':ep.get('repetition',0),'success':ep.get('success'),
            'request_n':len(requests),'checked_refs_n':len(checked),'N':metrics['N'],'X':metrics['X'],
            'NX_missing_reasons':nx_reasons,'zero_N':metrics['N']==0,'zero_X':metrics['X']==0,
            'trace_verified':not issues,'issues':issues,'capture':capture,'source_capture':source_capture,'unknown_execution_calls':unknown_calls,'retry_raw_paths':(ep.get('localization_sources') or {}).get('attempts',[])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit-dir',type=Path,default=PROFILING);args=ap.parse_args()
    manifest=load(RESULTS/'summary.json'); cells=[];history=[]
    for cell in manifest['cells']:
        if cell['model']!='deepseek-flash': continue
        path=str(relocate(cell['evaluation_path']).resolve()) if cell.get('evaluation_path') else None; result={'cell_id':cell['cell_id'],'evaluation_path':path,'total':cell['total'],'tasks':[],'cell_issues':[]}
        if path:
            evaluation=load(path); expected_files={}; identity_path=evaluation.get('provenance',{}).get('skill_identity')
            identity=load(identity_path) if identity_path else None
            result['version_id']=evaluation['version_id'];result['skill_identity_path']=identity_path
            if identity is None or identity.get('version_id')!=evaluation['version_id']:result['cell_issues'].append('missing_or_mismatched_skill_identity')
            summary=load(cell['summary_path']) if cell.get('summary_path') else None
            if summary:
                expected_files={'SKILL.md':summary['final_skill']}
                if version(summary['final_skill'])!=evaluation['version_id']: result['cell_issues'].append('final_skill_hash_mismatch')
            else:
                # Delivery contains original immutable skill text next to identity.json.
                if identity:
                    for name,digest in identity.get('files',{}).items():
                        base=relocate(identity_path).parent;candidates=[base/name,base/'files'/name,base/'native'/name]
                        for p in candidates:
                            if p.is_file(): expected_files[name]=p.read_text(); break
                        if name not in expected_files:result['cell_issues'].append('skill_text_file_not_located:'+name)
            if identity:
                for name,text in expected_files.items():
                    if hashlib.sha256(text.encode()).hexdigest()!=identity.get('files',{}).get(name):result['cell_issues'].append('delivery_file_hash_mismatch:'+name)
            evidence=Evidence();episodes={(e['task_id'],e.get('repetition',0)):e for e in evaluation['episodes']}
            overlay=recovery_overlay(evaluation,Path(path))
            for planned in evaluation['panel']:
                key=(planned['task_id'],planned.get('repetition',0)); ep=episodes.get(key)
                replacement=overlay.get(key)
                original_trace = audit_episode(ep,evidence,expected_files) if ep and replacement else None
                if replacement: ep=replacement[0]
                task=audit_episode(ep,evidence,expected_files) if ep else {'task_id':key[0],'repetition':key[1],'trace_verified':False,'issues':[{'kind':'planned_episode_missing','path':path}]}
                task['original_evaluation_path']=path
                task['remeasurement_evaluation_path']=replacement[1] if replacement else None
                task['effective_evaluation_path']=replacement[1] if replacement else path
                if original_trace:
                    task['original_trace_verified']=original_trace['trace_verified']
                    task['original_issues']=original_trace['issues']
                result['tasks'].append(task)
        else:result['cell_issues'].append('selected_test_evaluation_missing')
        for name, predicate in [('trace_verified_n',lambda t:t['trace_verified']),('known_n',lambda t:t.get('success') is not None),('N_known_n',lambda t:t.get('N') is not None),('X_known_n',lambda t:t.get('X') is not None),('zero_N_n',lambda t:t.get('zero_N')),('zero_X_n',lambda t:t.get('zero_X'))]:
            result[name]=sum(bool(predicate(t)) for t in result['tasks'])
        result['issue_counts']=dict(collections.Counter(i['kind'] for t in result['tasks'] for i in t['issues']))
        cells.append(result); print(result['cell_id'],result['trace_verified_n'],result['total'],result['issue_counts'],result['cell_issues'],flush=True)
        if cell['arm']=='skillopt':
            for p in sorted((relocate(cell['run_path']).resolve()/'results/evaluations/train').glob('*/evaluation.json')):
                e=load(p);unknown=[{'task_id':t['task_id'],'reason':t.get('outcome_reason'),'source_paths':list(refs(t))} for t in e['episodes'] if t.get('success') is None or t.get('outcome_valid') is not True]
                history.append({'cell_id':cell['cell_id'],'path':str(p),'episode_n':len(e['episodes']),'unknown_n':len(unknown),'unknown':unknown})
    out={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_generated_utc':manifest['generated_utc'],'manifest_path':str(RESULTS/'summary.json'),'semantics':'Offline physical file and JSON pointer audit. Zero N/X are estimator exclusions, not missing trajectories. Retry evidence is checked and preserved; no original data altered.','cells':cells,'skillopt_train_history':history}
    (args.audit_dir/'trace-integrity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# DeepSeek selected test trace integrity','',out['semantics'],'','| Cell | Verified / planned | Known outcome | N known | X known | zero N | zero X |','|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:lines.append(f"| {c['cell_id']} | {c['trace_verified_n']}/{c['total']} | {c['known_n']} | {c['N_known_n']} | {c['X_known_n']} | {c['zero_N_n']} | {c['zero_X_n']} |")
    lines+=['','Exact task-level gaps, evidence paths, and retained retry paths: [trace-integrity.json](trace-integrity.json).','']
    for c in cells:
        if c['cell_issues']:lines.append(c['cell_id']+': '+', '.join(c['cell_issues']))
        for t in c['tasks']:
            if t['issues']:lines.append(f"- {c['cell_id']} / {t['task_id']}: "+'; '.join(i['kind']+' '+str(i.get('path',i.get('reason',''))) for i in t['issues']))
    lines+=['',f"SkillOpt train history: {sum(h['unknown_n'] for h in history)} unknown/invalid episodes in {len(history)} evaluations. Review optimizer accept/reject decisions affected by unknown train outcomes before final claims; no history rewritten."]
    (args.audit_dir/'trace-integrity.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
