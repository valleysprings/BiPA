"""Offline policy/rollout guard checks; no model API or task execution."""
import ast
import json
from pathlib import Path
import tempfile
source=Path(__file__).resolve().parents[1] / 'adapters'/'skillopt.py'
tree=ast.parse(source.read_text())
names={'STRICT_POLICY','UnknownOutcome','outcome_policy','require_known'}
nodes=[n for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
ns={'Path':Path,'json':json};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
with tempfile.TemporaryDirectory() as d:
 root=Path(d)/'new';ns['outcome_policy'](root,True);ns['outcome_policy'](root,True)
 try:ns['outcome_policy'](root,False)
 except RuntimeError:pass
 else:raise AssertionError('legacy must not resume strict')
 old=Path(d)/'old';old.mkdir();(old/'runtime_state.json').write_text('{}')
 try:ns['outcome_policy'](old,True)
 except RuntimeError:pass
 else:raise AssertionError('strict must not adopt old checkpoint')
 before=(root/'outcome_policy.json').read_bytes()
 try:ns['require_known']([{'id':'a','success':1},{'id':'b','success':None}],root,'train','step_0001')
 except ns['UnknownOutcome']:pass
 else:raise AssertionError('must abort whole panel')
 assert (root/'outcome_policy.json').read_bytes()==before
 assert json.loads((root/'unknown_outcome.json').read_text())['requested']==2
 ns['require_known']([{'id':'b','success':0}],root,'dev','selection')
# Guard must precede score conversion/artifact creation in actual adapter.
rollout=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='rollout')
assert next(n.lineno for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require_known') < next(n.lineno for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='conversation')
print('strict policy and whole-panel abort offline checks passed')
# Real adapter and actual upstream train() integration, with evaluator mocked.
import sys
from unittest.mock import patch
base=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(base.parent),str(base/'optimizers/skillopt')]
from bipa.adapters import skillopt as arm
from types import SimpleNamespace
for failure_split in ['train','dev']:
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);seed=root/'seed.md';seed.write_text('seed')
  cfg=arm.flatten_config(arm.load_config(str(Path(arm.skillopt.__file__).parents[1]/'configs/_base_/default.yaml')))
  cfg.update(out_root=str(root),skill_init=str(seed),env='harness',optimizer_backend='openai_compatible',target_backend='openai_compatible',optimizer_model='offline',target_model='offline',train_size=1,batch_size=1,num_epochs=1,strict_outcomes=True)
  adapter=arm.Harness(SimpleNamespace(benchmark="offline"),{'train':['a'],'sel':['a'],'test':['a']},[['a']],cfg)
  seen=[]
  def evaluate(cell,text,ids,split):
   seen.append(split)
   return [{'id':'a','success':None if split==failure_split else 1,'messages':[],'verifier':None}]
  with patch.object(arm.executor,'evaluate',evaluate),patch.object(adapter,'reflect',side_effect=AssertionError('must not reflect unknown train')):
   try:arm.ReflACTTrainer(cfg,adapter).train()
   except arm.UnknownOutcome:pass
   else:raise AssertionError('upstream must propagate unknown abort')
  assert not (root/'history.json').exists()
  state=root/'runtime_state.json'
  if state.exists():assert json.loads(state.read_text())['last_completed_step']==0
  assert not (root/'steps/step_0001/step_record.json').exists()
  assert not list(root.glob('**/*sel*cache*'))
  assert failure_split in seen
print('real upstream trainer propagates train/baseline-dev abort without step commit')
