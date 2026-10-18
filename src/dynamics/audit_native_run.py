#!/usr/bin/env python3
"""Offline completion certificate, independent of optimizer/scheduler success flags."""
import argparse
import collections
import datetime
import hashlib
import json
import tempfile
from pathlib import Path

from src.dynamics.audit_trace_integrity import Evidence, audit_episode, refs
from src.core.layout import relocate


def request_payload_matches(source, record):
    """The capture exporter stores an absent response as {}; this is still missing evidence.

    Only normalize None and the empty dict. Completeness is checked independently;
    a populated or malformed response must never disappear behind this normalization.
    """
    def response(value):
        return {} if value is None else value
    return (source.get('request') == record.get('request')
            and response(source.get('response')) == response(record.get('response')))


def file_hash(path):
    return hashlib.sha256(relocate(path).read_bytes()).hexdigest()


def skill_version(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def slots(panel):
    return [(r, 0) if isinstance(r, str) else (r.get('task_id', r.get('id')), r.get('repetition', 0))
            for r in panel]


def ledger_segments(rows):
    """Native resume launches restart sequence numbering; audit each process segment."""
    segments=[];current=[]
    if rows and all(row.get('process_id') for row in rows):
        processes=collections.defaultdict(list)
        for row in rows:processes[row['process_id']].append(row)
        segments=list(processes.values())
    else:
        for row in rows:
            if row.get('state')=='capture_initialized':
                if current:segments.append(current)
                current=[]
            current.append(row)
        if current:segments.append(current)
    valid=bool(segments) and all(segment[0].get('state')=='capture_initialized'
        and all(row.get('sequence')==i and row.get('dropped_records')==0
                and row.get('protocol')=='native-v1' for i,row in enumerate(segment,1))
        for segment in segments)
    normal_exit=valid and all(segment[-1].get('state')=='capture_process_exit' for segment in segments)
    return valid, normal_exit


def native_timeout_cutoff(ep, ledger_path, protocol, witness):
    """A Harbor task timeout is a failure with a provable native teardown cutoff."""
    trial=relocate(ledger_path).parents[2]
    result_path,log_path=trial/'result.json',trial/'trial.log'
    if not result_path.is_file() or not log_path.is_file() or type(ep.get('execution_exit_code')) is not int:
        return None
    result=json.loads(result_path.read_text());exception=result.get('exception_info') or {}
    if exception.get('exception_type')!='AgentTimeoutError' or not exception.get('occurred_at'):
        return None
    agent_end=(result.get('agent_execution') or {}).get('finished_at');finished=result.get('finished_at')
    if not agent_end or not finished:return None
    if 'Warning: Agent environment cleanup failed' in log_path.read_text(errors='replace'):return None
    if (result.get('config',{}).get('environment') or {}).get('delete') is not True:return None
    semantic_sources=[]
    for tree in protocol.get('backend_code',{}).values():
        relative='vendor/harbor/src/harbor/trial/trial.py'
        digest=tree.get('files',{}).get(relative)
        code_path=Path(tree.get('root',''))/relative
        if not digest or not code_path.is_file() or file_hash(code_path)!=digest:continue
        code=code_path.read_text()
        # Frozen Harbor emits finished_at only after teardown; its file logger
        # captures DEBUG cleanup errors, so absence of the error has meaning.
        if ('file_handler.setLevel(logging.DEBUG)' in code and
            'await self._stop_agent_environment()\n        self.result.finished_at' in code and
            'Warning: Agent environment cleanup failed' in code):semantic_sources.append(code_path)
    if not semantic_sources:return None
    try:
        if datetime.datetime.fromisoformat(agent_end.replace('Z','+00:00'))>datetime.datetime.fromisoformat(finished.replace('Z','+00:00')):return None
    except ValueError:return None
    for path in (result_path,log_path,*semantic_sources):witness(path)
    return {'state':'aborted_at_native_timeout','agent_execution_finished_at':agent_end,
            'native_trial_finished_at':finished,'result_path':str(result_path),'trial_log_path':str(log_path),
            'execution_exit_code':ep['execution_exit_code'],
            'semantics':'Native failure; finalized after environment teardown under frozen DEBUG-logged Harbor implementation. No tool result synthesized.'}


def retry_attempt_scopes(path, evaluation, load):
    """Only the official retry lineage identifies unscored infrastructure attempts."""
    scopes={}; originals=[];issues=[]
    lineage=evaluation.get('unknown_outcome_retries') or {}
    if lineage and lineage.get('schema')!='bach-unknown-outcome-retries-v1':
        return scopes,originals,[{'kind':'retry_lineage_schema_invalid','path':str(path)}]
    attempts={a.get('attempt'):a for a in lineage.get('attempts',[])}
    effective={(e['task_id'],e.get('repetition',0)):e for e in evaluation.get('episodes',[])}
    for index,replaced in enumerate(lineage.get('replaced_episodes',[])):
        original=replaced.get('episode');key=slots([replaced.get('slot',{})])[0]
        selected=effective.get(key);number=replaced.get('recovered_by_attempt');chosen=attempts.get(number)
        selected_ref=(selected or {}).get('unknown_outcome_retry') or {}
        if (not original or original.get('success') is not None or (original.get('task_id'),original.get('repetition',0))!=key
            or key not in slots(lineage.get('initially_unknown',[])) or not selected or type(selected.get('success')) is not bool
            or not chosen or selected_ref.get('attempt')!=number
            or not any((r.get('task_id'),r.get('repetition',0))==key and r.get('success') is selected.get('success') for r in chosen.get('recovered',[]))
            or str(relocate(selected_ref.get('evaluation','')).resolve())!=str(relocate(chosen.get('evaluation','')).resolve())):
            issues.append({'kind':'retry_replacement_lineage_invalid','path':str(path),'index':index});continue
        selected_eval=load(chosen['evaluation']);selected_rows=[e for e in selected_eval.get('episodes',[]) if (e['task_id'],e.get('repetition',0))==key]
        if len(selected_rows)!=1 or {k:v for k,v in selected.items() if k!='unknown_outcome_retry'}!=selected_rows[0]:
            issues.append({'kind':'retry_selected_episode_mismatch','path':str(path),'index':index});continue
        originals.append((index,original))
        for n,attempt in attempts.items():
            if not isinstance(n,int) or n>=number or key not in slots(attempt.get('slots',[])):continue
            prior=load(attempt['evaluation']);rows=[e for e in prior.get('episodes',[]) if (e['task_id'],e.get('repetition',0))==key]
            if (len(rows)!=1 or rows[0].get('success') is not None or key not in slots(attempt.get('still_unknown',[]))):
                issues.append({'kind':'retry_prior_attempt_lineage_invalid','path':attempt['evaluation'],'slot':key});continue
            scopes[str(relocate(attempt['evaluation']).resolve()),key]={'effective_evaluation':str(path),'selected_evaluation':chosen['evaluation'],'selected_attempt':number,'attempt':n}
    return scopes,originals,issues


def infra_failure_proof(ep, evidence):
    """Unknown is not a verdict; require an actual structured failure and closed request map."""
    if ep.get('success') is not None or ep.get('outcome_reason') not in (
        'terminal_provider_failure','provider_stream_truncated','backend_infrastructure_failure','verifier_timeout'):
        return None
    sources=[]
    for request in ep.get('requests',[]):
        try:
            record=evidence.get(request['record_path']);source=evidence.get(record['source_ref'])
        except (ValueError,KeyError,IndexError,TypeError):return None
        sources.append((record['source_ref'],source))
    if not sources:return None
    ref,terminal=sources[-1];error=terminal.get('error') or (terminal.get('response') or {}).get('error')
    status=terminal.get('http_status')
    structured=(isinstance(error,dict) and error.get('type') not in (None,'response_unparseable','response_not_observed')) or (type(status) is int and status>=400) or terminal.get('response_event')=='response.failed'
    if not structured:return None
    # Every captured request must be represented; no disappearance behind retry success.
    mapped=collections.defaultdict(set)
    for source_ref,_ in sources:
        if '#normalized/model_calls/' not in source_ref:return None
        raw,index=source_ref.split('#normalized/model_calls/',1)
        try:mapped[raw].add(int(index))
        except ValueError:return None
    for raw,indexes in mapped.items():
        try: normalized=evidence.get(raw+'#normalized')
        except (ValueError,KeyError,IndexError,TypeError):return None
        if indexes!=set(range(len(normalized.get('model_calls',[])))):return None
    return {'state':'recorded_infrastructure_failure','terminal_source_ref':ref,'error':error,'http_status':status,
            'semantics':'Unscored infrastructure attempt linked by official retry lineage; no task verdict synthesized.'}


def certify(run):
    run = Path(run).resolve()
    issues = []
    evidence_paths = {}

    def witness(path):
        p = relocate(path).resolve()
        try:
            key = str(p.relative_to(run)) if p.is_relative_to(run) else str(p)
            evidence_paths[key] = file_hash(p)
        except OSError as exc:
            issues.append({'kind': 'evidence_file_unreadable', 'path': str(p), 'reason': str(exc)})

    def load(path):
        witness(path)
        try:
            return json.loads(relocate(path).read_text())
        except (OSError, ValueError) as exc:
            issues.append({'kind': 'json_unreadable', 'path': str(path), 'reason': str(exc)})
            return {}

    protocol_path, summary_path = run/'native_protocol.json', run/'summary.json'
    protocol, summary = load(protocol_path), load(summary_path)
    protocol_issues = []
    if protocol.get('protocol') != 'native-v1': protocol_issues.append('native_protocol_required')
    if protocol.get('feedback_mode') != 'full': protocol_issues.append('full_feedback_required')
    frozen = dict(protocol); recorded_hash = frozen.pop('config_hash', None)
    if not recorded_hash or skill_digest(frozen) != recorded_hash: protocol_issues.append('protocol_config_hash_mismatch')
    for k in ('arm','model','benchmark','harness','scale','config_hash'):
        if summary.get(k) != protocol.get(k): protocol_issues.append('summary_protocol_mismatch:'+k)
    if summary.get('protocol') != 'native-v1': protocol_issues.append('summary_native_protocol_required')
    issues.extend({'kind': x} for x in protocol_issues)

    event_paths = sorted((run/'round-evidence').glob('*.json'))
    events = [(p, load(p)) for p in event_paths]
    states = collections.defaultdict(list)
    evaluation_events = []
    for p, e in events:
        if e.get('arm') != protocol.get('arm'):
            issues.append({'kind':'round_arm_mismatch','path':str(p)})
        if e.get('kind') == 'round_state': states[e.get('step')].append((p,e))
        elif e.get('kind') == 'round_evaluation': evaluation_events.append((p,e))
    step_issues=[]; steps=[]
    optimizer = protocol.get('arm') in ('skillopt','gepa','skilladam')
    expected_steps = len(protocol.get('batches', [])) if optimizer else 0
    original_events={}
    if optimizer:
        from src.dynamics.audit_skill_history import Store, skillopt, gepa, skilladam
        try:
            with tempfile.TemporaryDirectory(prefix='native-state-audit-') as directory:
                original,_,_={'skillopt':skillopt,'gepa':gepa,'skilladam':skilladam}[protocol['arm']](run,Store(Path(directory)))
                original_events={e['step']:e for e in original if e.get('kind')=='optimizer_step'}
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            step_issues.append({'kind':'original_optimizer_log_unreadable','reason':str(exc)})
    # All optimizers are audited against their frozen schedule, including skipped rounds.
    for step in range(1, expected_steps+1):
        matches=states.get(step,[])
        if len(matches)!=1:
            step_issues.append({'kind':'round_state_missing_or_ambiguous','step':step,'count':len(matches)})
            continue
        p,e=matches[0]; local=[]
        original=original_events.get(step)
        if original is None:local.append('original_optimizer_round_missing')
        else:
            if e.get('decision')!=original.get('decision'):local.append('original_optimizer_decision_mismatch')
            for state_key,original_key in (('input','before'),('candidate','proposed'),('returned','after')):
                if (e.get(state_key) or {}).get('sha256')!=original.get(original_key,{}).get('sha256'):
                    local.append('original_optimizer_'+state_key+'_mismatch')
            if e.get('parent_candidate_idx')!=original.get('parent_candidate_idx'):
                local.append('original_optimizer_parent_mismatch')
        if not e.get('source_files'):local.append('optimizer_state_source_missing')
        for source,digest in e.get('source_files',{}).items():
            witness(source)
            try:
                if file_hash(source)!=digest:local.append('optimizer_state_source_changed')
            except OSError:local.append('optimizer_state_source_unreadable')
        for name in ('input','candidate','returned'):
            identity=e.get(name)
            if identity is None:
                if name!='candidate': local.append('missing_'+name)
                continue
            text=identity.get('text')
            if not isinstance(text,str) or hashlib.sha256(text.encode()).hexdigest()!=identity.get('sha256') or skill_version({'SKILL.md':text})!=identity.get('version_id'):
                local.append(name+'_hash_mismatch')
        if not e.get('decision'):local.append('decision_missing')
        if e.get('candidate') is None and e.get('decision') not in ('skip_perfect','reflection_failed','no_proposal','skip_no_patches','skip_no_edits','skip'):
            local.append('candidate_absent_without_skip_witness')
        if e.get('candidate') is None and e.get('input') != e.get('returned'):
            local.append('skip_changed_incumbent')
        if e.get('decision')=='reject' and e.get('input')!=e.get('returned'):local.append('rejection_changed_incumbent')
        if str(e.get('decision')).startswith('accept') or e.get('decision')=='force_accept':
            if e.get('candidate')!=e.get('returned'):local.append('accepted_candidate_return_mismatch')
        for ref in e.get('evaluation_records',[]):
            linked=load(ref)
            if linked.get('kind')!='round_evaluation' or linked.get('step')!=step:
                local.append('round_evaluation_link_mismatch')
        # A state cannot silently certify a round without its original evaluation witness.
        if not any(x.get('step')==step for _,x in evaluation_events): local.append('round_evaluation_witness_missing')
        steps.append({'step':step,'path':str(p),'decision':e.get('decision'),'issues':local})
        step_issues.extend({'kind':x,'step':step,'path':str(p)} for x in local)
    step_objects_verified = not step_issues and (not optimizer or len(steps)==expected_steps)
    issues.extend(step_issues)

    evaluations=[]; engine=Evidence(); evaluated={}
    paths=sorted((run/'results/evaluations').rglob('evaluation.json'))
    if not paths: issues.append({'kind':'evaluations_missing'})
    snapshots={relocate(event['evaluation_path']).resolve() for _,event in evaluation_events
               if event.get('effective_evaluation_path') and event.get('evaluation_path')}
    paths=sorted({p.resolve() for p in paths} | snapshots)
    documents={str(path.resolve()):load(path) for path in paths};infra_scopes={};extra=[]
    for path in paths:
        e=documents[str(path.resolve())]
        scoped,originals,lineage_issues=retry_attempt_scopes(path,e,load)
        infra_scopes.update(scoped);issues.extend(lineage_issues)
        for index,original in originals:
            shadow={**e,'panel':[{'task_id':original['task_id'],'repetition':original.get('repetition',0)}],'episodes':[original]}
            label=str(path)+'#unknown_outcome_retries/replaced_episodes/'+str(index)
            extra.append((label,shadow,{'effective_evaluation':str(path),'attempt':0}))
    work=[(str(path),documents[str(path.resolve())],None) for path in paths]+extra
    for path,e,shadow_scope in work:
        local=[]
        identity_path=e.get('provenance',{}).get('skill_identity')
        identity=load(identity_path) if identity_path else {}
        if identity.get('version_id')!=e.get('version_id'):local.append('skill_identity_version_mismatch')
        files={}
        identity_files=identity.get('files',{})
        if not isinstance(identity_files,dict):
            local.append('skill_identity_files_schema_invalid');identity_files={}
        for name,digest in identity_files.items():
            if not identity_path:break
            base=relocate(identity_path).parent;candidates=[base/name,base/'files'/name,base/'native'/name]
            actual=next((p for p in candidates if p.is_file()),None)
            if actual is None:local.append('skill_file_missing:'+name);continue
            witness(actual);files[name]=actual.read_text()
            if file_hash(actual)!=digest:local.append('skill_file_hash_mismatch:'+name)
        if skill_version(files)!=e.get('version_id'):local.append('skill_version_mismatch')
        panel=e.get('panel',[]); planned=slots(panel); eps=collections.defaultdict(list)
        for ep in e.get('episodes',[]):eps[(ep.get('task_id'),ep.get('repetition',0))].append(ep)
        if len(set(planned))!=len(planned):local.append('duplicate_panel_slot')
        if set(eps)-set(planned):local.append('unplanned_episode_slot')
        tasks=[]
        for key in planned:
            rows=eps.get(key,[])
            if len(rows)!=1:
                tasks.append({'task_id':key[0],'repetition':key[1],'trace_verified':False,'issues':[{'kind':'planned_episode_missing_or_duplicate','count':len(rows)}]});continue
            ep=rows[0]
            scope=shadow_scope or infra_scopes.get((str(relocate(path).resolve()),key))
            try: task=audit_episode(ep,engine,files)
            except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
                task={'task_id':key[0],'repetition':key[1],'trace_verified':False,'issues':[{'kind':'trace_audit_exception','reason':str(exc)}]}
            if (ep.get('capture') or {}).get('tool_results_complete') is False:
                task['issues'].append({'kind':'tool_results_incomplete'})
            if not ep.get('requests'):task['issues'].append({'kind':'model_request_evidence_missing'})
            if not ep.get('evidence_refs') or not (ep.get('localization_sources') or {}).get('results'):
                task['issues'].append({'kind':'native_outcome_source_missing'})
            backend_outcome=ep.get('backend_outcome') or {}
            if not scope and backend_outcome.get('success') is not ep.get('success'):
                task['issues'].append({'kind':'backend_outcome_success_mismatch'})
            ledger_paths={r.split('#',1)[0] for r in refs(ep) if 'tool-execution.jsonl' in r}
            dispatch_ids=set();returned_ids=set();timeout_cutoffs=[]
            if protocol.get('harness')=='opencode' and not ledger_paths:
                task['issues'].append({'kind':'native_dispatch_ledger_missing'})
            for ledger_path in ledger_paths:
                witness(ledger_path)
                try:
                    ledger=[json.loads(line) for line in relocate(ledger_path).read_text().splitlines() if line.strip()]
                    dispatch_ids.update(row.get('call_id') for row in ledger if row.get('state')=='entered_dispatch')
                    returned_ids.update(row.get('call_id') for row in ledger if row.get('state')=='completed')
                    valid,normal_exit=ledger_segments(ledger)
                    cutoff=native_timeout_cutoff(ep,ledger_path,protocol,witness) if not normal_exit or not dispatch_ids<=returned_ids else None
                    if cutoff:
                        end=datetime.datetime.fromisoformat(cutoff['native_trial_finished_at'].replace('Z','+00:00'))
                        for row in ledger:
                            value=row.get('time')
                            if not value or datetime.datetime.fromisoformat(value.replace('Z','+00:00'))>end:
                                cutoff=None;break
                    if cutoff:timeout_cutoffs.append(cutoff)
                    if not valid:task['issues'].append({'kind':'native_dispatch_ledger_coverage_invalid','path':ledger_path})
                    if not normal_exit and not cutoff:
                        task['issues'].append({'kind':'native_process_exit_witness_missing','path':ledger_path})
                    log_paths=list(relocate(ledger_path).parent.parent.glob('opencode*.txt'))
                    if not log_paths:task['issues'].append({'kind':'native_capture_stderr_missing','path':ledger_path})
                    for log_path in log_paths:
                        witness(log_path)
                        if 'HARNESS_NATIVE_LEDGER_WRITE_FAILED' in log_path.read_text(errors='replace'):
                            task['issues'].append({'kind':'native_dispatch_ledger_write_failed','path':str(log_path)})
                except (OSError,ValueError,TypeError,AttributeError,IndexError) as exc:
                    task['issues'].append({'kind':'native_dispatch_ledger_unreadable','path':ledger_path,'reason':str(exc)})
            response_call_ids=set()
            for req in ep.get('requests',[]):
                try: record=engine.get(req.get('record_path'))
                except (ValueError,KeyError,TypeError,IndexError):continue
                try:
                    source=engine.get(record.get('source_ref'))
                    if not request_payload_matches(source, record):
                        task['issues'].append({'kind':'request_source_payload_mismatch','request_id':req.get('request_id')})
                except (ValueError,KeyError,TypeError,IndexError,AttributeError):
                    task['issues'].append({'kind':'request_source_payload_unresolved','request_id':req.get('request_id')})
                response=record.get('response') or {}
                native_calls={c.get('id'):c.get('function',{}) for choice in response.get('choices',[]) for c in choice.get('message',{}).get('tool_calls',[])}
                native_calls.update({c.get('call_id'):c for c in response.get('output',[]) if c.get('type') in ('function_call','custom_tool_call')})
                for call in req.get('tool_calls',[]):
                    call_id=str(call.get('call_id'));prefix=str(req.get('request_id'))+':'
                    source_id=call_id[len(prefix):] if call_id.startswith(prefix) else call_id
                    response_call_ids.add(source_id)
                    original=native_calls.get(source_id)
                    if original is None or original.get('name')!=call.get('tool'):
                        task['issues'].append({'kind':'tool_call_source_mismatch','call_id':call.get('call_id')})
                    if call.get('executed') is True and not call.get('evidence_refs'):
                        task['issues'].append({'kind':'tool_dispatch_evidence_missing','call_id':call.get('call_id')})
                    if call.get('execution_state')=='entered_dispatch' and not timeout_cutoffs:
                        task['issues'].append({'kind':'dispatch_without_terminal_result','call_id':call.get('call_id')})
            if ledger_paths and dispatch_ids!=response_call_ids:
                task['issues'].append({'kind':'native_dispatch_call_id_closure_mismatch',
                    'dispatch_only':sorted(dispatch_ids-response_call_ids),'response_only':sorted(response_call_ids-dispatch_ids)})
            if timeout_cutoffs:
                task['native_termination_cutoffs']=timeout_cutoffs
                task['unfinished_tool_calls']=[{'call_id':call_id,'state':'aborted_at_native_timeout'} for call_id in sorted(dispatch_ids-returned_ids)]
                task['issues']=[i for i in task['issues'] if i['kind']!='tool_results_incomplete']
            if scope:
                proof=infra_failure_proof(ep,engine)
                task['retry_lineage']=scope
                if proof:
                    task['infrastructure_failure_evidence']=proof
                    task['algorithm_q_eligible']=False
                    task['issues']=[i for i in task['issues'] if i['kind']!='outcome_unknown_or_invalid']
                else:task['issues'].append({'kind':'infrastructure_abort_evidence_incomplete'})
            if ledger_paths and dispatch_ids==response_call_ids and (dispatch_ids<=returned_ids or timeout_cutoffs):
                # Recompute from physical witnesses, including legacy multi-resume ledgers.
                # Preserve the original flag; never rewrite the evaluated artifact.
                other=[i for i in task['issues'] if i['kind']!='invocations_complete']
                if not other:
                    task['original_invocations_complete']=(ep.get('capture') or {}).get('invocations_complete')
                    task['invocations_verified_from_physical_ledger']=True
                    task['issues']=other
            task['trace_verified']=not task['issues'];tasks.append(task)
        item={'path':str(path),'split':e.get('split'),'version_id':e.get('version_id'),'panel':panel,
              'issues':local,'tasks':tasks,'verified':not local and bool(planned) and all(t['trace_verified'] for t in tasks)}
        evaluations.append(item)
        if not shadow_scope:evaluated[str(relocate(path).resolve())]=e
    for key in engine.cache:
        if not key.endswith('#normalized'):witness(key)
    for p,event in evaluation_events:
        ref=event.get('evaluation_path'); actual=evaluated.get(str(relocate(ref).resolve())) if ref else None
        if actual is None or file_hash(ref)!=event.get('evaluation_sha256'):
            issues.append({'kind':'round_evaluation_file_missing_or_changed','path':str(p)})
        elif actual.get('panel')!=event.get('panel') or actual.get('split')!=event.get('split'):
            issues.append({'kind':'round_evaluation_panel_mismatch','path':str(p)})
        identity=event.get('skill');expected=skill_version({} if identity is None else {'SKILL.md':identity.get('text')})
        if actual and actual.get('version_id')!=expected:issues.append({'kind':'round_evaluation_skill_mismatch','path':str(p)})
    final_files={} if summary.get('final_skill') is None else {'SKILL.md':summary.get('final_skill')}
    final_path=run/'final/SKILL.md';witness(final_path)
    if not final_path.is_file() or final_path.read_text()!=(summary.get('final_skill') or ''):issues.append({'kind':'final_skill_summary_mismatch'})
    final_version=skill_version(final_files);test_panel=slots(protocol.get('panels',{}).get('test',[]))
    final_tests=[x for x in evaluations if x['split']=='test' and x['version_id']==final_version and slots(x['panel'])==test_panel]
    if not test_panel or not final_tests:issues.append({'kind':'frozen_final_test_panel_missing'})
    if protocol.get('arm')=='seed':
        for panel_key,split in (('train','train'),('sel','dev'),('test','test')):
            frozen_panel=slots(protocol.get('panels',{}).get(panel_key,[]))
            if not frozen_panel or not any(x['split']==split and x['version_id']==final_version and slots(x['panel'])==frozen_panel for x in evaluations):
                issues.append({'kind':'frozen_baseline_panel_missing','split':split})
    all_traces=bool(evaluations) and all(x['verified'] for x in evaluations)
    full_trace=all_traces and not any(i['kind'].startswith('round_evaluation_') for i in issues)
    complete=step_objects_verified and full_trace and not issues
    return {'kind':'native_completion_certificate','schema_version':1,'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_path':str(run),'protocol_sha256':file_hash(protocol_path) if protocol_path.is_file() else None,
            'summary_sha256':file_hash(summary_path) if summary_path.is_file() else None,
            'step_objects_verified':step_objects_verified,'all_eval_task_traces_verified':all_traces,
            'full_trace_verified':full_trace,'complete':complete,'issues':issues,'steps':steps,
            'evaluations':evaluations,'evidence_paths':evidence_paths,
            'semantics':'Original physical evidence only. Known outcomes and capture flags alone never certify full trajectories. Before-only dispatch may witness N but is not a completed trajectory.'}


def skill_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();out=args.output or args.run/'completion_audit.json'
    try:
        result=certify(args.run)
    except Exception as exc:
        result={'kind':'native_completion_certificate','schema_version':1,'run_path':str(args.run.resolve()),
                'protocol_sha256':None,'summary_sha256':None,'step_objects_verified':False,
                'all_eval_task_traces_verified':False,'full_trace_verified':False,'complete':False,
                'issues':[{'kind':'certificate_audit_exception','exception_type':type(exc).__name__,'reason':str(exc)}],
                'evidence_paths':{}}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('run_path','step_objects_verified','all_eval_task_traces_verified','full_trace_verified','complete')}))


if __name__=='__main__':main()
