"""Reproducible trigger timing / metric-confounding and dense Gramian cost audit.
No source file from the previous revision is modified. Not a quantum resource benchmark.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python trigger_cost_audit.py
"""
from pathlib import Path
import sys, json, time, csv
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits
OLD = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OLD))
from VQA_main_validation import build_problem
from modal_lowrank import weighted_matrix, calc_flow, tangent, prj, score_choice

def simulate(L,y0,dt,T,ranking_mode,trigger_mode,budget,thr,W,reference):
    norm0=la.norm(y0); phi=y0/norm0; logm=np.log(norm0)
    pairs=[1]; selected=[]; times=np.arange(int(round(T/dt))+1)*dt
    hist=np.zeros((len(times),len(y0))); hist[0]=y0
    fired_at=[]; mismatched_at=[]; n_checks=0; n_chosen=0
    t0=time.perf_counter()
    for i,t in enumerate(times[:-1]):
        while len(pairs)<budget:
            r_e,s_e=score_choice(phi,L,pairs,W,'euclidean')
            r_w,s_w=score_choice(phi,L,pairs,W,'transient')
            n_checks+=1
            if (r_e>thr)!=(r_w>thr):mismatched_at.append(float(t))
            trigger = r_e if trigger_mode=='euclidean' else r_w
            scores = s_e if ranking_mode=='euclidean' else s_w
            if trigger<=thr or not scores or scores[0][0]<1e-14: break
            k=int(scores[0][1]);pairs.append(k)
            n_chosen+=1
            fired_at.append(float(t))
            selected.append(dict(time=float(t),pair=k,trigger_e=float(r_e),trigger_w=float(r_w),ranker=ranking_mode,trigger=trigger_mode))
        def rhs(v):
            v=v/la.norm(v);return prj(calc_flow(v,L),tangent(v,pairs)),float(v@L@v)
        k1,a1=rhs(phi);v=phi+dt/2*k1;v/=la.norm(v)
        k2,a2=rhs(v);v=phi+dt/2*k2;v/=la.norm(v)
        k3,a3=rhs(v);v=phi+dt*k3;v/=la.norm(v)
        k4,a4=rhs(v)
        phi+=dt/6*(k1+2*k2+2*k3+k4);phi/=la.norm(phi)
        logm+=dt/6*(a1+2*a2+2*a3+a4)
        hist[i+1]=np.exp(logm)*phi
    elapsed=time.perf_counter()-t0
    err=la.norm(hist-reference,axis=1)/la.norm(reference,axis=1)
    return dict(ranking=ranking_mode,trigger=trigger_mode,chosen_pairs=pairs,max_error=float(err.max()),final_error=float(err[-1]),
        selections=selected,selection_times=fired_at,number_of_comparisons=n_checks,metric_trigger_disagreements=len(mismatched_at),first_metric_disagreement=mismatched_at[0] if mismatched_at else None,number_of_enrichments=n_chosen,wall_seconds=round(elapsed,6))

def run(n,dt):
    a0=.10;beta=.25;D=.05;gam=.05;T=1.;tau=.30;thr=.005;budget=8
    t0=time.perf_counter();prob=build_problem(n,a0,beta,D,gam,lambda_bc=5)
    pdiff=build_problem(n,0.,0.,D,0.,lambda_bc=5)
    t1=time.perf_counter();eigs,V=la.eigh(-pdiff['L']); A=V.T@prob['L']@V; y0=V.T@prob['y0']
    if y0[0]<0:V[:,0]*=-1;A=V.T@prob['L']@V;y0=V.T@prob['y0']
    t2=time.perf_counter();W=weighted_matrix(A,tau);t3=time.perf_counter()
    times=np.arange(int(round(T/dt))+1)*dt
    reference=expm_multiply(A,y0,start=0,stop=T,num=len(times));t4=time.perf_counter()
    out=[]
    for rank,trig in [('euclidean','euclidean'),('transient','transient'),('transient','euclidean')]:
        out.append(simulate(A,y0,dt,T,rank,trig,budget,thr,W,reference))
    return dict(n=n,dt=dt,ambient_real_tangent_rank=n-1,budget=budget,tau=tau,threshold=thr,
        timings_seconds=dict(assembly=t1-t0,modal_eigh=t2-t1,gramian_40_expm=t3-t2,reference=t4-t3),
        gramian_quadrature_nodes=40,matrix_exponentials=40,dense_gramian_products=40,
        experimental_runs=out)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        res=[run(32,.002),run(64,.0005)]
    path=ROOT/'results'/'trigger_cost_audit.json';path.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
