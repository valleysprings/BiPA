#!/usr/bin/env python3
"""Derive evidence-linked skill history without changing optimizer state or calling APIs."""
import argparse, collections, datetime, hashlib, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bipa.layout import (BASELINE_ROOT, BASELINES_ROOT, MITIGATION_ROOT as ROOT,
                    ARTIFACTS, RESULTS, EXECUTION, ATTRIBUTION, PROFILING, CACHE, OPTIMIZERS, relocate)

def read(p): return json.loads(relocate(p).read_text())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def ver(s): return sha(json.dumps({'SKILL.md':s},sort_keys=True,ensure_ascii=False,separators=(',',':')))
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
class Store:
 def __init__(self,p):self.p=p
 def text(self,text,source,reason=None):
  if text is None:return {'status':'unknown','reason':reason or 'not uniquely witnessed','sources':source}
  h=sha(text);p=self.p/'snapshots'/f'{h}.md';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
  return {'status':'known','sha256':h,'version_id':ver(text),'snapshot_path':str(p),'sources':source}
 def file(self,p):return self.text(p.read_text(),[str(p)]) if p.exists() else self.text(None,[str(p)],'file absent')
def contamination(run):
 out=[]
 for p in sorted((run/'results/evaluations').glob('*/*/evaluation.json')):
  e=read(p)
  if e.get('split') not in ('train','dev'):continue
  bad=[{'task_id':x['task_id'],'outcome_reason':x.get('outcome_reason')} for x in e['episodes'] if x.get('success') is None]
  if bad:out.append({'evaluation_path':str(p),'version_id':e['version_id'],'split':e['split'],'unknown_tasks':bad})
 return out

def evaluation_evidence(run, events):
 """Check original panels and physical traces; version matches are not temporal links."""
 from audit_trace_integrity import Evidence, audit_episode
 by_version=collections.defaultdict(list)
 expected={r['version_id']:relocate(r['snapshot_path']).read_text()
           for event in events for k in ('before','proposed','after','best')
           for r in [event.get(k,{})] if r.get('status')=='known'}
 for p in sorted((run/'results/evaluations').glob('*/*/evaluation.json')):
  e=read(p)
  if e.get('split') not in ('train','dev'):continue
  issues=[];identity_path=e.get('provenance',{}).get('skill_identity')
  identity=read(identity_path) if identity_path and relocate(identity_path).is_file() else None
  if not identity or identity.get('version_id')!=e['version_id']:issues.append('missing_or_mismatched_skill_identity')
  skill=expected.get(e['version_id']);files={'SKILL.md':skill} if skill is not None else {}
  if skill is None:issues.append('skill_text_not_linked_to_round')
  elif identity and identity.get('files',{}).get('SKILL.md')!=sha(skill):issues.append('skill_identity_hash_mismatch')
  episodes=collections.defaultdict(list)
  for ep in e['episodes']:episodes[(ep['task_id'],ep.get('repetition',0))].append(ep)
  evidence=Evidence();tasks=[]
  for planned in e['panel']:
   key=(planned['task_id'],planned.get('repetition',0));matches=episodes.get(key,[])
   if len(matches)!=1:
    tasks.append({'task_id':key[0],'repetition':key[1],'trace_verified':False,'issues':[{'kind':'planned_episode_missing_or_duplicate','count':len(matches)}]})
   else:tasks.append(audit_episode(matches[0],evidence,files))
  planned_keys={(x['task_id'],x.get('repetition',0)) for x in e['panel']}
  if len(planned_keys)!=len(e['panel']):issues.append('duplicate_panel_slot')
  if set(episodes)-planned_keys:issues.append('unplanned_episode_slot')
  by_version[e['version_id']].append({'evaluation_path':str(p),'split':e['split'],'skill_identity_path':identity_path,'panel':e['panel'],'planned_n':len(e['panel']),'episode_n':len(e['episodes']),'trace_verified_n':sum(t['trace_verified'] for t in tasks),'cell_issues':issues,'tasks':tasks})
 for event in events:
  event['evaluation_evidence']={k:[{f:e[f] for f in ('evaluation_path','split','planned_n','episode_n','trace_verified_n','cell_issues')} for e in by_version.get(event.get(k,{}).get('version_id'),[])] for k in ('before','proposed','after')}
  event['evaluation_link_semantics']='Content version match only; no round execution or score attribution inferred. Original evidence only; no remeasurement overlays.'
 return by_version

def skillopt(run,store):
 p=run/'skillopt';history=read(p/'history.json');events=[];seed=store.file(p/'skills/skill_v0000.md');before=seed;best=seed
 for d in sorted((p/'steps').glob('step_*')):
  step=int(d.name.split('_')[-1]);h=next((x for x in history if x['step']==step),None)
  proposal=store.file(d/'candidate_skill.md');decision=h.get('action') if h else 'pending'
  if h and h.get('candidate_hash') and proposal['status']=='known':
   if proposal['sha256'][:16]!=h['candidate_hash']:raise ValueError(f'candidate hash mismatch {d}')
  after=proposal if decision.startswith('accept') or decision=='force_accept' else before if h else store.text(None,[str(d)],'pending decision')
  if h and h.get('best_origin')==f'step_{step:04d}':best=proposal
  event={'step':step,'kind':'optimizer_step','decision':decision,'before':before,'proposed':proposal,'after':after,'best':best,'parent_sha256':before.get('sha256'),'sources':[str(d/'step_record.json'),str(p/'history.json')],'history_record':h}
  events.append(event)
  if h:before=after
 for d in sorted((p/'slow_update').glob('epoch_*')):
  result=read(d/'slow_result.json') if (d/'slow_result.json').exists() else {}
  runtime=read(p/'runtime_state.json');current=store.file(relocate(runtime['current_skill_path']))
  events.append({'kind':'epoch_slow_update','epoch':result.get('epoch'),'decision':result.get('action','unknown'),'before':before,'proposed':store.text(None,[str(d)],'placeholder event has no optimizer proposal'),'after':current,'best':store.file(p/'best_skill.md'),'sources':[str(d/'slow_result.json'),str(p/'runtime_state.json')],'parent_sha256':before.get('sha256')})
 return events,{'best':store.file(p/'best_skill.md'),'final':store.file(run/'final/SKILL.md')},sum(x.get('candidate_hash') is not None for x in history)

def gepa_block_starts(lines, step, record):
 """Resolve resumed text logs only when exactly one completed block matches the persisted decision.

 The upstream JSON checkpoint has one committed record; text logs also retain aborted attempts.
 Ambiguous completed attempts remain unauditable rather than choosing the last occurrence.
 """
 starts=[i for i,l in enumerate(lines) if l.startswith(f'Iteration {step}: Selected program ')]
 if len(starts)<=1:return starts
 matches=[]
 for start in starts:
  parent=re.match(rf'Iteration {step}: Selected program (\d+) ',lines[start])
  if not parent or int(parent[1])!=record.get('selected_program_candidate'):continue
  end=next((i for i in range(start+1,len(lines)) if re.match(r'Iteration \d+: Selected program ',lines[i])),len(lines))
  block=''.join(lines[start:end]);newidx=record.get('new_program_idx')
  if newidx is not None:
   committed=re.search(rf'^Iteration {step}: New program candidate index: {newidx}\s*$',block,re.M)
  else:
   committed=any(marker in block for marker in ('not better than old score','All subsample scores perfect',
                                               'Traceback (most recent call last)','did not propose'))
  if committed:matches.append(start)
 return matches


def gepa(run,store):
 p=run/'gepa';logs=read(p/'run_log.json');pool=read(p/'candidates.json');text=(p/'run_log.txt').read_text();lines=text.splitlines(keepends=True);events=[];mapped=0
 calls=[]
 for f in sorted((run/'llm').glob('call-*.json')):
  c=read(f);body=((c.get('response')or{}).get('choices')or[{}])[0].get('message',{}).get('content')
  if isinstance(body,str):calls.append((str(f),body))
 poolrefs=[store.text(x['current_candidate'],[f'{p}/candidates.json#{i}/current_candidate']) for i,x in enumerate(pool)]
 best_so_far=poolrefs[0] if poolrefs and text.startswith('Iteration 0: Base program full valset score:') else store.text(None,[str(p/'run_log.txt')],'initial best not witnessed')
 for record in logs:
  n=record['i']+1;starts=gepa_block_starts(lines,n,record)
  if len(starts)!=1:
   events.append({'step':n,'decision':'unknown','reason':'iteration log boundary ambiguous','record':record});continue
  start=starts[0];end=next((i for i in range(start+1,len(lines)) if re.match(r'Iteration \d+: Selected program ',lines[i])),len(lines));block=''.join(lines[start:end]);parent=record.get('selected_program_candidate');before=poolrefs[parent] if isinstance(parent,int) and parent<len(poolrefs) else store.text(None,[str(p/'run_log.json')])
  proposal=store.text(None,[str(p/'run_log.txt')],'no candidate proposed');idx=next((i for i in range(start,end) if lines[i].startswith(f'Iteration {n}: Proposed new text for current_candidate: ')),None)
  if idx is not None:
   stop=next((i for i in range(idx+1,end) if lines[i].startswith(f'Iteration {n}: ')),end)
   raw=lines[idx].split('Proposed new text for current_candidate: ',1)[1]+''.join(lines[idx+1:stop]);raw=raw[:-1] if raw.endswith('\n') else raw
   matches=[(f,b) for f,b in calls if raw in b]
   sources=[f'{p}/run_log.txt:lines:{idx+1}-{stop}']+[f+'#response/choices/0/message/content' for f,b in matches]
   if len(matches)==1:proposal=store.text(raw,sources);mapped+=1
   else:proposal=store.text(None,sources,f'LLM text correspondence not unique: {len(matches)} matches')
  newidx=record.get('new_program_idx');decision='accept' if newidx is not None else 'reject' if 'not better than old score' in block else 'skip_perfect' if 'All subsample scores perfect' in block else 'reflection_failed' if 'Traceback (most recent call last)' in block else 'no_proposal' if 'did not propose' in block else 'unknown'
  after=poolrefs[newidx] if isinstance(newidx,int) and newidx<len(poolrefs) else before if decision in ('reject','skip_perfect','reflection_failed','no_proposal') else store.text(None,[str(p/'run_log.json')])
  if decision=='accept' and proposal['status']=='known' and proposal['sha256']!=after.get('sha256'):raise ValueError(f'GEPA accepted proposal/pool mismatch {run} {n}')
  # Only explicit per-iteration best-index lines establish historical best.
  bm=re.findall(rf'Iteration {n}: Best program as per aggregate score on valset: (\d+)',block)
  best=poolrefs[int(bm[-1])] if bm and int(bm[-1])<len(poolrefs) else best_so_far if decision in ('reject','skip_perfect','reflection_failed','no_proposal') else store.text(None,[str(p/'run_log.txt')],'accepted candidate without unique best-selection witness')
  best_so_far=best
  events.append({'step':n,'iteration_id':record.get('iteration_id'),'kind':'optimizer_step','decision':decision,'before':before,'proposed':proposal,'after':after,'best':best,'parent_candidate_idx':parent,'parent_sha256':before.get('sha256'),'new_candidate_idx':newidx,'sources':[f'{p}/run_log.json#{record["i"]}',f'{p}/run_log.txt:lines:{start+1}-{end}'],'record':record,'failure_evidence':[l.strip() for l in block.splitlines() if 'Error:' in l or 'Traceback' in l]})
 return events,{'final':store.file(run/'final/SKILL.md'),'pool':poolrefs},mapped

def skilladam_rounds(p):
 """Completed iterations from upstream's checkpoint and persisted patches, the candidate rebuilt by upstream's own
 applier on the incumbent (FeedbackLoop._generate_candidate); round 0 is the adapter's seed.md."""
 sys.path.insert(0,str(OPTIMIZERS/'skilladam'))
 from skilladam.core.diff_apply import apply_patch_with_fallback
 checkpoint=read(p/'checkpoint.json');before=relocate(p/'seed.md').read_text();rounds=[]
 if sha(before)!=checkpoint['run_signature']['post_decision']['initial_skill_sha256']:raise ValueError(f'SkillAdam seed hash mismatch {p}')
 for r in checkpoint['iterations']:
  path=p/f"iterations/iter_{r['iteration']:03d}/patch_attempt_{r['patch_attempts']:02d}.json";patch=read(path)['patch']
  proposed=None if r['patch_mode']=='failed' else apply_patch_with_fallback(before,patch).text;after=proposed if r['accepted'] else before
  rounds.append({'record':r,'before':before,'patch':patch,'patch_path':str(path),'proposed':proposed,'after':after});before=after
 if before!=checkpoint['current_skill']:raise ValueError(f'SkillAdam rebuilt incumbent differs from checkpoint {p}')
 return rounds

def skilladam(run,store):
 p=run/'skilladam';rounds=skilladam_rounds(p);events=[];evaluated=collections.defaultdict(set)
 for f in sorted((run/'round-evidence').glob('*.json')):
  e=read(f)
  if e.get('kind')=='round_evaluation' and e.get('arm')=='skilladam' and e.get('role')=='validation_candidate':evaluated[e['step']].add((e.get('skill') or {}).get('sha256'))
 for x in rounds:
  r=x['record'];step=r['iteration']+1;sources=[f"{p/'checkpoint.json'}#iterations/{r['iteration']}",x['patch_path']]
  if x['proposed'] is not None and evaluated[step] and sha(x['proposed']) not in evaluated[step]:raise ValueError(f'SkillAdam rebuilt candidate differs from its evaluation {run} {step}')
  before,after=store.text(x['before'],sources),store.text(x['after'],sources)
  decision='accept' if r['accepted'] else 'no_proposal' if r['patch_mode']=='failed' else 'reject'
  events.append({'step':step,'kind':'optimizer_step','decision':decision,'before':before,'proposed':store.text(x['proposed'],sources,'patch application failed: no candidate'),'after':after,'best':after,'parent_sha256':before.get('sha256'),'sources':sources,'record':r})
 return events,{'final':store.file(run/'final/SKILL.md'),'current':store.file(p/'current_skill.md')},sum(x['proposed'] is not None for x in rounds)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=RESULTS/'skill-history');args=ap.parse_args();manifest=read(RESULTS/'summary.json');cells=[]
 for c in manifest['cells']:
  if c['model']!='deepseek-flash' or c['arm'] not in ('skillopt','gepa','skilladam'):continue
  run=relocate(c['run_path']).resolve();folder=args.output_dir/c['arm']/c['benchmark'];store=Store(folder)
  events,final,validated={'skillopt':skillopt,'gepa':gepa,'skilladam':skilladam}[c['arm']](run,store)
  evaluations=evaluation_evidence(run,events)
  bad=contamination(run)
  validity_path=PROFILING/'sequential-validity.json'
  boundaries=read(validity_path).get('cells',[]) if validity_path.exists() else []
  boundary=next((b for b in boundaries if b.get('arm')==c['arm'] and b.get('benchmark')==c['benchmark'] and b.get('run_path') and relocate(b['run_path']).resolve()==run.resolve()),None)
  first=boundary.get('first_affected_step') if boundary else None
  for e in events:
   hashes={e.get(k,{}).get('version_id') for k in ('before','proposed')}
   impact=[b for b in bad if b['version_id'] in hashes]
   e['unlocated_version_evidence']=impact
   e['sequential_validity_status']=('confirmed affected boundary or subsequent dependent step' if e.get('step') is None or e['step']>=first else 'before confirmed affected boundary') if boundary else 'not established'
   if boundary:e['sequential_validity_evidence']={'path':str(validity_path),'boundary':boundary}
  record={'cell_id':c['cell_id'],'run_path':str(run),'events':events,'evaluation_evidence_by_version':evaluations,'final_selection':final,'validated_candidate_n':validated,'decision_counts':dict(collections.Counter(e['decision'] for e in events)),'unknown_evaluations':bad,'first_confirmed_affected_step':first,'semantics':'Derived snapshots preserve witnessed text. Unknown fields never synthesized. Version-only evidence is unlocated and never establishes a first affected step; confirmed temporal boundaries come only from sequential-validity.json. GEPA after is selected lineage child/parent, not global pool best.'}
  write(folder/'index.json',record);cells.append({k:v for k,v in record.items() if k not in ('events','evaluation_evidence_by_version','final_selection','unknown_evaluations')});cells[-1]['index_path']=str(folder/'index.json')
 out={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'DeepSeek SkillOpt, GEPA and SkillAdam cells; native has no optimization rounds','cells':cells};write(args.output_dir/'summary.json',out)
 lines=['# Skill history evidence index','',out['scope'],'','All source files remain unchanged. Derived snapshots are content-addressed; unknown means no unique witness. Epoch slow-update events are separate from gate steps.','', '| Cell | Validated proposals | Decisions | First confirmed affected step |','|---|---:|---|---:|']
 for c in cells:lines.append(f"| [{c['cell_id']}]({c['index_path']}) | {c['validated_candidate_n']} | {c['decision_counts']} | {c['first_confirmed_affected_step']} |")
 (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
