"""Physical resolution and scale-aware non-normality check for smooth regime."""
from pathlib import Path
import json,sys
import numpy as np,scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parent))
from VQA_main_validation import build_problem
ROOT=Path(__file__).resolve().parents[1]
a0=.1;beta=.25;D=.05;gamma=.05
vals={};rows=[]
for n in [16,32,64,128]:
    P=build_problem(n,a0,beta,D,gamma,5);L=P['L'];
    yT=expm_multiply(L,P['y0'],start=0,stop=1,num=2)[-1];z=P['M_inv_half']@yT;fullz=np.r_[0,z,0]
    vals[n]=(P['x'],fullz,P['M_full'])
    comm=L@L.T-L.T@L
    mu=la.eigvalsh((L+L.T)/2)[-1];alpha=max(la.eigvals(L).real)
    # For n<=64, dense propagation transient operator bound on coarse grid only.
    opgrow=None
    if n<=64:opgrow=max(la.svdvals(la.expm(t*L))[0] for t in np.linspace(0,1,21))
    row={'n':n,'max_Pe':(abs(a0)+beta/2)/(2*D*(n+1)),'alpha':float(alpha),'mu2':float(mu),'scaled_comm':float(la.norm(comm,2)/la.norm(L,2)**2),'operator_norm_max_t0_1':opgrow,'reference_growth_end':float(la.norm(yT)/la.norm(P['y0']))}
    rows.append(row)
for nlo,nhi in [(16,32),(32,64),(64,128)]:
    xf,zf,Mf=vals[nhi];xc,zc,_=vals[nlo]
    zi=np.interp(xf,xc,zc)
    rel=np.sqrt((zf-zi)@Mf@(zf-zi))/np.sqrt(zf@Mf@zf)
    for r in rows:
        if r['n']==nlo:r['relative_fem_ref_difference_next_mesh_t1']=float(rel)
print(json.dumps(rows,indent=2));(ROOT/'results'/'mesh_resolution_new.json').write_text(json.dumps(rows,indent=2))
