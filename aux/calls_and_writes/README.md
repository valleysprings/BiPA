# calls_and_writes

Tool calls per turn and the read / write class of each AutoBench call: how the serialize rules put writes one per turn, and what packing the calls as densely as the seed would save.
`python -m aux.calls_and_writes.callsperturn` writes `callsperturn.json` and `mech_callsperturn.md` (calls per turn, single-read turns, turns saved by merging them).
`python -m aux.calls_and_writes.calls_rw extract`, then `stats`, then `python -m aux.calls_and_writes.boot_cf` and `python -m aux.calls_and_writes.write_record` give `mech_calls_rw.md` (writes sharing a turn, the packing counterfactual and its intervals). `calls_rw_data.pkl` and `calls_rw.json` under `$BIPA_OUT/calls_and_writes` are the data of the serialized-writes figure.
