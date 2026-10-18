import json

from src.dynamics.audit_native_run import certify, skill_digest, ledger_segments, native_timeout_cutoff, file_hash
from src.dynamics.audit_trace_integrity import Evidence


def test_empty_run_cannot_be_complete(tmp_path):
    result = certify(tmp_path)
    assert not result['complete']
    assert not result['all_eval_task_traces_verified']
    assert result['protocol_sha256'] is None


def test_summary_score_cannot_replace_original_traces(tmp_path):
    protocol = {'protocol':'native-v1','feedback_mode':'full','arm':'none',
                'model':'m','benchmark':'b','harness':'opencode','scale':'smoke',
                'batches':[], 'panels':{'test':['one']}}
    protocol['config_hash'] = skill_digest(protocol)
    (tmp_path/'native_protocol.json').write_text(json.dumps(protocol))
    summary = {**{k:protocol[k] for k in ('protocol','arm','model','benchmark','harness','scale','config_hash')},
               'final_skill':None,'test':{'final':{'known':1,'tasks':1,'success':1}}}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'final').mkdir()
    (tmp_path/'final/SKILL.md').write_text('')
    result = certify(tmp_path)
    assert result['step_objects_verified']
    assert not result['complete']
    assert any(i['kind']=='frozen_final_test_panel_missing' for i in result['issues'])
    assert 'summary.json' in result['evidence_paths']


def test_native_jsonl_line_pointer_is_one_based(tmp_path):
    path=tmp_path/'tool-execution.jsonl'
    path.write_text('{"state":"capture_initialized"}\n{"state":"entered_dispatch","call_id":"abc"}\n')
    evidence=Evidence()
    assert evidence.get(str(path)+'#line=2/call_id')=='abc'
    try:
        evidence.get(str(path)+'#line=0')
    except ValueError:
        pass
    else:
        raise AssertionError('zero-based pointer accepted')


def test_resumed_native_processes_have_independent_sequences():
    segment=[{'state':state,'sequence':i,'dropped_records':0,'protocol':'native-v1'}
             for i,state in enumerate(('capture_initialized','entered_dispatch','completed','capture_process_exit'),1)]
    assert ledger_segments(segment+segment)==(True,True)
    broken=[dict(row) for row in segment]
    broken[2]['sequence']=4
    assert ledger_segments(segment+broken)==(False,False)
    assert ledger_segments(segment[:-1])==(True,False)


def test_timeout_requires_frozen_teardown_and_cleanup_evidence(tmp_path):
    ledger=tmp_path/'trial/agent/opencode/tool-execution.jsonl'
    ledger.parent.mkdir(parents=True)
    trial=ledger.parents[2]
    result={'exception_info':{'exception_type':'AgentTimeoutError','occurred_at':'2026-10-03T00:00:01Z'},
            'agent_execution':{'finished_at':'2026-10-03T00:00:02Z'},'finished_at':'2026-10-03T00:00:03Z',
            'config':{'environment':{'delete':True}}}
    (trial/'result.json').write_text(json.dumps(result));(trial/'trial.log').write_text('debug logs')
    relative='vendor/harbor/src/harbor/trial/trial.py';code=tmp_path/relative;code.parent.mkdir(parents=True)
    code.write_text('file_handler.setLevel(logging.DEBUG)\nawait self._stop_agent_environment()\n        self.result.finished_at\nWarning: Agent environment cleanup failed')
    protocol={'backend_code':{'worker':{'root':str(tmp_path),'files':{relative:file_hash(code)}}}}
    witness=[]
    assert native_timeout_cutoff({'execution_exit_code':0},ledger,protocol,witness.append)['state']=='aborted_at_native_timeout'
    assert len(witness)==3
    (trial/'trial.log').write_text('Warning: Agent environment cleanup failed')
    assert native_timeout_cutoff({'execution_exit_code':0},ledger,protocol,witness.append) is None


def retry_run(tmp_path, missing_error=False, prior_known_false=False):
    version=skill_digest({});directory=tmp_path/'results/evaluations/test/v';directory.mkdir(parents=True)
    identity=tmp_path/'identity.json';identity.write_text(json.dumps({'version_id':version,'files':{}}))
    def episode(label,success):
        raw=tmp_path/(label+'-raw.json');record=tmp_path/(label+'-record.json');outcome=tmp_path/(label+'-outcome.json')
        call={'request':{'input':[]},'response':{'choices':[]} if success is not None else None}
        if success is None and not missing_error:call['error']={'type':'HTTPError','message':'503 unavailable'};call['http_status']=503
        raw.write_text(json.dumps({'model_calls':[call],'extra':{'capture_status':'complete'}}));outcome.write_text('{}')
        record.write_text(json.dumps({'request_id':label,'source_ref':str(raw)+'#normalized/model_calls/0',
            'request':call['request'],'response':call['response'],'usage':{'input_tokens':0}}))
        return {'task_id':'one','repetition':0,'success':success,'outcome_valid':success is not None,
            'outcome_reason':'terminal_provider_failure' if success is None else 'task_verdict',
            'backend_outcome':{'success':success},'requests':[{'request_id':label,'record_path':str(record),'input_tokens':0,'tool_calls':[]}],
            'capture':{'requests_complete':True,'invocations_complete':True,'scored_attempt':str(raw)},
            'skill_delivery':{'verified':True,'files':{}},'evidence_refs':[str(raw),str(outcome)],
            'localization_sources':{'results':str(outcome)},'execution_exit_code':0}
    first=episode('first',None);second=episode('second',False if prior_known_false else None);last=episode('last',True)
    def evaluation(ep):return {'version_id':version,'split':'test','panel':[{'task_id':'one','repetition':0}],
        'episodes':[ep],'provenance':{'skill_identity':str(identity)}}
    retries=[]
    for number,ep in ((1,second),(2,last)):
        folder=directory/f'retry-{number}';folder.mkdir();path=folder/'evaluation.json';path.write_text(json.dumps(evaluation(ep)))
        retries.append({'attempt':number,'evaluation':str(path),'slots':[{'task_id':'one','repetition':0}],
            'still_unknown':[{'task_id':'one','repetition':0}] if number==1 else [],'recovered':[{'task_id':'one','repetition':0,'success':True}] if number==2 else []})
    selected={**last,'unknown_outcome_retry':{'attempt':2,'evaluation':retries[1]['evaluation']}}
    base=evaluation(selected);base['unknown_outcome_retries']={'schema':'bach-unknown-outcome-retries-v1','initially_unknown':[{'task_id':'one','repetition':0}],'attempts':retries,'replaced_episodes':[
        {'slot':{'task_id':'one','repetition':0},'attempt':0,'recovered_by_attempt':2,'episode':first}]}
    (directory/'evaluation.json').write_text(json.dumps(base))
    protocol={'protocol':'native-v1','feedback_mode':'full','arm':'none','model':'m','benchmark':'b','harness':'fixture',
        'scale':'smoke','batches':[],'panels':{'test':['one']}}
    protocol['config_hash']=skill_digest(protocol);(tmp_path/'native_protocol.json').write_text(json.dumps(protocol))
    summary={**{k:protocol[k] for k in ('protocol','arm','model','benchmark','harness','scale','config_hash')},'final_skill':None}
    (tmp_path/'summary.json').write_text(json.dumps(summary));(tmp_path/'final').mkdir();(tmp_path/'final/SKILL.md').write_text('')


def test_official_unknown_unknown_known_retry_preserves_aborted_evidence(tmp_path):
    retry_run(tmp_path)
    result=certify(tmp_path)
    assert result['complete'], result
    aborted=[t for e in result['evaluations'] for t in e['tasks'] if t.get('algorithm_q_eligible') is False]
    assert len(aborted)==2
    assert all(t['success'] is None and t['trace_verified'] for t in aborted)


def test_audit_reads_immutable_round_snapshot_after_live_metadata_changes(tmp_path):
    from src.native_trace import record_round_evaluation
    retry_run(tmp_path)
    path = tmp_path / 'results/evaluations/test/v/evaluation.json'
    record_round_evaluation(tmp_path, arm='none', step=0, role='final_test',
                            skill_text=None, evaluation_path=path)
    updated = json.loads(path.read_text())
    updated['metadata_note'] = 'effective evaluation updated after the round witness'
    path.write_text(json.dumps(updated))
    result = certify(tmp_path)
    assert result['complete'], result['issues']
    assert any('/round-evidence/evaluations/' in item['path'] for item in result['evaluations'])


def test_retry_success_cannot_hide_missing_failure_evidence(tmp_path):
    retry_run(tmp_path,missing_error=True)
    result=certify(tmp_path)
    assert not result['complete']
    assert any(i['kind']=='infrastructure_abort_evidence_incomplete' for e in result['evaluations'] for t in e['tasks'] for i in t['issues'])


def test_known_false_cannot_be_exempted_as_infrastructure(tmp_path):
    retry_run(tmp_path,prior_known_false=True)
    result=certify(tmp_path)
    assert not result['complete']
    assert any(i['kind']=='retry_prior_attempt_lineage_invalid' for i in result['issues'])
