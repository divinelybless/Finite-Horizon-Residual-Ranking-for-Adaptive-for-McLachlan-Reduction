"""Resolved smooth-flow sub-full-tangent benchmark; noiseless classical prototype.
Modal coordinates require a classical global transform: NOT a cheap quantum state-preparation protocol.
All parameter settings and obtained results written to results/.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
from scipy.special import roots_legendre
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'code'))
from VQA_main_validation import build_problem, spd_square_root_and_inverse
OUT=ROOT/'results'; FIG=ROOT/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

def weighted_matrix(L, tau):
    # Gauss-Legendre numerical integration of finite-horizon observability Gramian.
    x,w=roots_legendre(40); W=np.zeros_like(L)
    for xi,wi in zip(x,w):
        s=tau*(xi+1)/2; E=la.expm(s*L)
        W+=(tau/2)*wi*(E.T@E)
    return (W+W.T)/2

def prj(vec, K):
    if K.shape[1]==0:return np.zeros_like(vec)
    return K@la.lstsq(K, vec,cond=1e-10)[0]

def calc_flow(phi,L):
    return L@phi-(phi@L@phi)*phi

def tangent(phi, pairs):
    K=np.zeros((len(phi),len(pairs)))
    for j,k in enumerate(pairs):K[0,j]=-phi[k]; K[k,j]=phi[0]
    return K

def score_choice(phi,L,pairs,W,mode):
    K=tangent(phi,pairs); G=calc_flow(phi,L)
    if mode=='euclidean':U=np.eye(len(phi)); F=np.eye(len(phi))
    else:U=W;F=la.cholesky(W,lower=False)
    # weighted projection residual with metric W; Tikhonov free
    VK=F@K;VG=F@G
    q=la.lstsq(VK, VG,cond=1e-10)[0] if pairs else np.zeros(0)
    r=VG-VK@q
    rel=la.norm(r)/max(la.norm(VG),1e-15)
    scores=[]
    for k in range(1,len(phi)):
        if k in pairs:continue
        col=np.zeros_like(phi);col[0]=-phi[k];col[k]=phi[0]
        v=F@col
        if pairs:v-=VK@la.lstsq(VK,v,cond=1e-10)[0]
        den=v@v
        if den>1e-12: scores.append(((v@r)**2/den,k))
    scores.sort(key=lambda z:(-z[0],z[1]))
    return rel,scores

def sim(L,y0,dt,T,mode,tau,budget,thr,W):
    norm0=la.norm(y0);phi=y0/norm0;logm=np.log(norm0)
    pairs=[1] # star (0,1) initial pair
    times=np.arange(int(round(T/dt))+1)*dt
    hist=np.zeros((len(times),len(y0)));hist[0]=y0
    selected=[];ranks=np.zeros(len(times),dtype=int)
    for i,t in enumerate(times[:-1]):
        # repeated enrichment at same time node, no step-count induced limit
        while len(pairs)<budget:
            rel,scores=score_choice(phi,L,pairs,W,mode)
            if rel<=thr or not scores or scores[0][0]<1e-14:break
            pairs.append(scores[0][1]);selected.append([t,scores[0][1],scores[0][0],rel]);
        K=tangent(phi,pairs);ranks[i]=np.linalg.matrix_rank(K,tol=1e-10)
        def rhs(v):
            v=v/la.norm(v);return prj(calc_flow(v,L),tangent(v,pairs)),float(v@L@v)
        k1,a1=rhs(phi);v=phi+dt/2*k1;v/=la.norm(v)
        k2,a2=rhs(v);v=phi+dt/2*k2;v/=la.norm(v)
        k3,a3=rhs(v);v=phi+dt*k3;v/=la.norm(v)
        k4,a4=rhs(v)
        phi+=dt/6*(k1+2*k2+2*k3+k4);phi/=la.norm(phi)
        logm+=dt/6*(a1+2*a2+2*a3+a4)
        hist[i+1]=np.exp(logm)*phi
    ranks[-1]=np.linalg.matrix_rank(tangent(phi,pairs),tol=1e-10)
    return times,hist,pairs,np.array(selected).reshape(-1,4),ranks

def main():
    n=int(os.getenv("VQA_N","32")); D=float(os.getenv("VQA_D",".02")); a0=float(os.getenv("VQA_A",".25"));beta=float(os.getenv("VQA_B",".8"));gam=.05;dt=float(os.getenv("VQA_DT",".002"));T=float(os.getenv("VQA_T","1.0"));tau=float(os.getenv("VQA_TAU",".25"))
    prob=build_problem(n,a0,beta,D,gam,lambda_bc=5)
    # Real symmetric FEM diffusion stiffness K_D from same mesh, eigen-decomposition in whitened mass geometry
    pdiff=build_problem(n,0.,0.,D,0.,lambda_bc=5)
    eigs,V=la.eigh(-pdiff['L']); A=V.T@prob['L']@V
    y0=V.T@prob['y0'];
    if y0[0]<0:V[:,0]*=-1;A=V.T@prob['L']@V;y0=V.T@prob['y0']
    # Orient other modal vectors consistently? sign doesn't change exact dynamics.
    W=weighted_matrix(A,tau)
    times=np.arange(round(T/dt)+1)*dt
    reference=expm_multiply(A,y0,start=0,stop=T,num=len(times))
    rows=[]; saved={}
    for budget in [int(v) for v in os.getenv("VQA_BUDGETS","3,5,8,12").split(",")]:
        for mode in ['euclidean','transient']:
            t,h,p,selected,ranks=sim(A,y0,dt,T,mode,tau,budget,thr=.005,W=W)
            errors=la.norm(h-reference,axis=1)/la.norm(reference,axis=1)
            record=dict(n=n,Pe_max=(abs(a0)+beta/2)/(2*D*(n+1)),resolution_scale=np.sqrt(D/beta)*(n+1),budget=budget,method=mode,max_error=float(errors.max()),final_error=float(errors[-1]),num_active=len(p),tangent_rank=int(ranks[-1]),selected=list(map(int,p)),tau=tau,dt=dt)
            rows.append(record);saved[(budget,mode)]=(errors,selected)
            print('MODE',budget,mode,'MAX',errors.max(),'FINAL',errors[-1],'PAIRS',p,'rank',ranks[-1],flush=True)
    with (OUT/'modal_lowrank_results.json').open('w') as f:json.dump(rows,f,indent=2)
    np.savez_compressed(OUT/'modal_lowrank_trajectories.npz',times=times,reference=reference, **{f'e_{b}_{m}':v[0] for (b,m),v in saved.items()})
    plt.figure(figsize=(6.8,4.25));
    for b in [b for b in [5,8,12] if (b,"euclidean") in saved]:
        for m in ['euclidean','transient']:
            e=saved[(b,m)][0];plt.plot(times,e,label=f'{m}, p={b}',lw=1.5,ls='--' if m=='transient' else '-')
    plt.yscale('log');plt.xlabel('t');plt.ylabel('relative error');plt.legend(fontsize=8,ncol=2);plt.grid(alpha=.2);plt.tight_layout();plt.savefig(FIG/'resolved_modal_accuracy.pdf');plt.close()
    with (OUT/'modal_setup.json').open('w') as f:json.dump({'n':n,'a0':a0,'beta':beta,'D':D,'gamma':gam,'dt':dt,'T':T,'tau':tau,'threshold':.005,'coordinate_basis':'mass-whitened Dirichlet diffusion modes','candidate_pool':'(0,j) for 1<=j<n','enrichment':'repeat per step until threshold or budget','full_tangent_rank':n-1,'no_hardware':'true'},f,indent=2)
if __name__=='__main__':main()
