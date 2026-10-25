# calls_and_writes

Tool calls per turn, the read / write class of each AutoBench call, serialized writes and the packing counterfactual, with the bip_writes figure, the share of writes issued alongside another write and the turns saved by packing.
`python -m aux.calls_and_writes.callsperturn [draw]` writes `callsperturn.json` and `mech_callsperturn.md` (or draws `mech_calls`).
`python -m aux.calls_and_writes.calls_rw extract`, then `stats`, then `python -m aux.calls_and_writes.boot_cf` and `python -m aux.calls_and_writes.write_record` give `mech_calls_rw.md`.
`python -m aux.calls_and_writes.fig_writes` draws `bip_writes` from `calls_rw_data.pkl` and `calls_rw.json` (outputs under `$BIPA_OUT/calls_and_writes` and `$BIPA_OUT/token_attribution`).
