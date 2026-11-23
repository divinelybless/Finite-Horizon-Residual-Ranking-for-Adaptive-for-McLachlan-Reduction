"""Verify that the claims in the revised main manuscript match archived run files."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]/'results'
read=lambda name:json.loads((root/name).read_text())
def find(rows,n,p,mode):return next(z for z in rows if z['n']==n and z['budget']==p and z['method']==mode)
main=read('modal_n32_dt002.json');fine=read('modal_n64_dt0005.json');half=read('modal_n32_dt001.json');growth=read('modal_n32_growth_case.json')
assert abs(find(main,32,8,'euclidean')['max_error'] - 0.0344723444)<1e-8
assert abs(find(main,32,8,'transient')['max_error'] - 0.0024229258)<1e-8
assert abs(find(fine,64,8,'transient')['max_error'] - 0.0026415641)<1e-8
assert abs(find(main,32,8,'transient')['max_error']-find(half,32,8,'transient')['max_error'])<1e-6
assert abs(find(growth,32,8,'euclidean')['max_error']-find(growth,32,8,'transient')['max_error'])<1e-10
fixed=read('gate_product_results.json')
adapt=read('adaptive_gate_product_results.json')
assert abs(next(z for z in fixed if z['p']==5)['max_relative_error']-.00254828455)<1e-8
assert abs(next(z for z in adapt if z['p_max']==5 and z['method']=='euclidean')['max_error']-.0316509863)<1e-7
print('PASS: 7 primary claims match saved numerical results')
