"""Isolated single-thread dense setup/ranking timing, not quantum hardware timing.
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python score_timing.py
"""
from pathlib import Path
import json,time,sys,platform
import numpy as np
import scipy
import scipy.linalg as la
from threadpoolctl import threadpool_limits,threadpool_info
OLD=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OLD))
from VQA_main_validation import build_problem
from modal_lowrank import weighted_matrix, score_choice

def med_timing(f,k=9):
    a=[]
    for i in range(k):
        start=time.perf_counter(); f();a.append(time.perf_counter()-start)
    return dict(median_s=float(np.median(a)), min_s=float(min(a)), max_s=float(max(a)), runs=k)

def run(n):
    pdiff=build_problem(n,0,0,.05,0,lambda_bc=5)
    p=build_problem(n,.10,.25,.05,.05,lambda_bc=5)
    vals,V=la.eigh(-pdiff['L']);A=V.T@p['L']@V;y0=V.T@p['y0']
    if y0[0]<0: V[:,0]*=-1;A=V.T@p['L']@V;y0=V.T@p['y0']
    phi=y0/la.norm(y0)
    W=weighted_matrix(A,.30)
    seqE=[1,3,4,5,6,7,8,9]
    seqW=[1,3,4,5,6,7,2,8]
    def eval_seq(mode,seq):
        for i in range(1,len(seq)):
            ratio,scores=score_choice(phi,A,seq[:i],W,mode)
    tW=med_timing(lambda:weighted_matrix(A,.30))
    tEig=med_timing(lambda:la.eigh(-pdiff['L']))
    tE=med_timing(lambda:eval_seq('euclidean',seqE))
    tR=med_timing(lambda:eval_seq('transient',seqW))
    return dict(n=n,quadrature_nodes=40,dense_expm_calls=40,dense_gramian_products=40,
                full_dense_diffusion_eigh=1,scoring_decisions=7,
                evaluated_candidates=sum((n-1-i) for i in range(1,8)),
                metric_cholesky_fact_count=7,
                reference_hardware='single-threaded local container, illustrative only',
                eigensolver=tEig,gramian=tW,euclidean_seven_scores=tE,transient_seven_scores=tR)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        data=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threadpool=threadpool_info(),runs=[run(32),run(64)])
    (ROOT/'results'/'dense_cost_timing.json').write_text(json.dumps(data,indent=2))
    for r in data['runs']:
        print(r['n'], [(s,round(r[s]['median_s']*1000,3)) for s in ['eigensolver','gramian','euclidean_seven_scores','transient_seven_scores']])
