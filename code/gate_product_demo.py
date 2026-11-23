"""Actual ordered product of real two-level rotations with analytic parameter tangents.
Classical statevector emulation (no hardware compilation or shots)."""
import json, sys
from pathlib import Path
import numpy as np, scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from VQA_main_validation import build_problem
OUT=ROOT/'results';FIG=ROOT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)

def rotate(v,pair,theta):
    i,j=pair;c=np.cos(theta);s=np.sin(theta);w=v.copy()
    w[i]=c*v[i]-s*v[j];w[j]=s*v[i]+c*v[j]
    return w

def K(v,pair):
    i,j=pair;w=np.zeros_like(v);w[i]=-v[j];w[j]=v[i];return w

def state_jacobian(theta,phi0,pairs):
    states=[phi0]
    for pair,angle in zip(pairs,theta):states.append(rotate(states[-1],pair,angle))
    columns=[]
    for i,pair in enumerate(pairs):
        v=K(states[i+1],pair)
        for j in range(i+1,len(pairs)):v=rotate(v,pairs[j],theta[j])
        columns.append(v)
    return states[-1],np.column_stack(columns)

def execute(n,p,dt=.002,T=1.0):
    # Resolved moderate advection: same coefficients as resolved modal study, but smaller n for gate demonstration.
    a0=.1;beta=.25;D=.05;gamma=.05
    prob=build_problem(n,a0,beta,D,gamma,5.0)
    pure=build_problem(n,0.,0.,D,0.,5.0)
    vals,U=la.eigh(-pure['L']);A=U.T@prob['L']@U;y0=U.T@prob['y0'];r0=la.norm(y0);phi0=y0/r0
    pairs=[(0,j) for j in range(1,p+1)];theta=np.zeros(p);logr=np.log(r0)
    t=np.arange(round(T/dt)+1)*dt;trajectory=np.zeros((len(t),n));trajectory[0]=y0
    cond_list=[];velocity_err=[]
    for k in range(len(t)-1):
        def rhs(x):
            th=x[:p];phi,J=state_jacobian(th,phi0,pairs)
            G=A@phi-(phi@A@phi)*phi
            dth=la.lstsq(J,G,cond=1e-10)[0]
            return np.r_[dth,float(phi@A@phi)]
        x=np.r_[theta,logr];k1=rhs(x);k2=rhs(x+dt/2*k1);k3=rhs(x+dt/2*k2);k4=rhs(x+dt*k3)
        x=x+dt/6*(k1+2*k2+2*k3+k4);theta=x[:p];logr=x[-1]
        phi,J=state_jacobian(theta,phi0,pairs)
        trajectory[k+1]=np.exp(logr)*phi
        cond_list.append(np.linalg.cond(J))
    ref=expm_multiply(A,y0,start=0,stop=T,num=len(t));err=la.norm(trajectory-ref,axis=1)/la.norm(ref,axis=1)
    # Analytic Jacobian is checked independently at an off-zero random theta.
    testtheta=np.linspace(.01,.03,p);st,J=state_jacobian(testtheta,phi0,pairs)
    h=1e-6
    fd=np.column_stack([(state_jacobian(testtheta+np.eye(p)[j]*h,phi0,pairs)[0]-state_jacobian(testtheta-np.eye(p)[j]*h,phi0,pairs)[0])/(2*h) for j in range(p)])
    derivative_error=la.norm(J-fd)
    result=dict(n=n,qubits=int(np.log2(n)),p=p,full_real_tangent_rank=n-1,dt=dt,T=T,max_relative_error=float(err.max()),final_relative_error=float(err[-1]),analytic_tangent_finite_difference_error=float(derivative_error),max_gram_condition=float(np.max(cond_list)),max_norm_deviation=float(max(abs(la.norm(state_jacobian(np.zeros(p),phi0,pairs)[0])-1),abs(la.norm(st)-1))),generator_pairs=pairs,ansatz='R_p(theta_p)...R_1(theta_1)phi0',parameters_integrated=True,measurement_shots=0)
    print(result,flush=True)
    return t,err,result

def main():
    plots=[];rows=[]
    for p in [3,5,6]:
        t,e,row=execute(8,p);plots.append((p,t,e));rows.append(row)
    (OUT/'gate_product_results.json').write_text(json.dumps(rows,indent=2))
    plt.figure(figsize=(6.6,3.9))
    for p,t,e in plots:plt.plot(t,e,label=f'{p} real Givens gates')
    plt.xlabel('t');plt.ylabel('relative semi-discrete error');plt.yscale('log');plt.grid(alpha=.2);plt.legend();plt.tight_layout();plt.savefig(FIG/'gate_product_test.pdf');plt.close()
if __name__=='__main__':main()

# Additional circuit-level adaptive experiment: each new Givens factor is
# prepended on the LEFT at zero angle; the exact product Jacobian is then used.
def execute_adaptive(mode='euclidean', n=8, budget=5, threshold=.005, dt=.002, T=1.0, tau=.3):
    from modal_lowrank import weighted_matrix
    D=.05;a0=.1;beta=.25;gamma=.05
    prob=build_problem(n,a0,beta,D,gamma,5)
    pure=build_problem(n,0,0,D,0,5)
    _,U=la.eigh(-pure['L']);A=U.T@prob['L']@U
    y0=U.T@prob['y0'];r0=la.norm(y0);phi0=y0/r0
    W=weighted_matrix(A,tau);F=la.cholesky(W,lower=False) if mode=='transient' else np.eye(n)
    pairs=[(0,1)]; theta=np.zeros(1);logr=np.log(r0)
    times=np.arange(round(T/dt)+1)*dt;history=np.zeros((len(times),n));history[0]=y0;events=[]
    for step,t in enumerate(times[:-1]):
        while len(pairs)<budget:
            phi,J=state_jacobian(theta,phi0,pairs)
            G=A@phi-(phi@A@phi)*phi; V=F@J;g=F@G
            projected=V@la.lstsq(V,g,cond=1e-10)[0];rel=la.norm(g-projected)/max(la.norm(g),1e-14)
            if rel<=threshold:break
            candidates=[]
            for j in range(1,n):
                if (0,j) in pairs:continue
                c=F@K(phi,(0,j));r=c-V@la.lstsq(V,c,cond=1e-10)[0];den=r@r
                if den>1e-12: candidates.append(((r@(g-projected))**2/den,j))
            if not candidates:break
            score,j=sorted(candidates,key=lambda z:(-z[0],z[1]))[0]
            if score<=1e-14:break
            pairs.append((0,j));theta=np.r_[theta,0.0];events.append((float(t),int(j),float(score)))
        m=len(pairs)
        def rhs(x):
            phi,J=state_jacobian(x[:m],phi0,pairs);G=A@phi-(phi@A@phi)*phi
            return np.r_[la.lstsq(J,G,cond=1e-10)[0],float(phi@A@phi)]
        x=np.r_[theta,logr];k1=rhs(x);k2=rhs(x+dt/2*k1);k3=rhs(x+dt/2*k2);k4=rhs(x+dt*k3)
        x+=dt/6*(k1+2*k2+2*k3+k4);theta=x[:m];logr=x[-1]
        history[step+1]=np.exp(logr)*state_jacobian(theta,phi0,pairs)[0]
    ref=expm_multiply(A,y0,start=0,stop=T,num=len(times))
    err=la.norm(history-ref,axis=1)/la.norm(ref,axis=1)
    row={'n':n,'method':mode,'p_max':budget,'p_final':len(pairs),'pairs':pairs,'enrichment_events':events,'max_error':float(err.max()),'final_error':float(err[-1]),'threshold':threshold,'tau':tau,'integrator':'parameter RK4','orientation':'left-prepend new factor at zero angle'}
    return times,err,row

if __name__=='__main__':
    out=[]; plt.figure(figsize=(6.6,4))
    for budget in (3,5):
        for mode in ('euclidean','transient'):
            t,err,row=execute_adaptive(mode,budget=budget);print('CIRCUIT ADAPTIVE',budget,mode,row,flush=True)
            out.append(row);plt.plot(t,err,label=f'{mode} p={budget}',ls='--' if mode=='transient' else '-')
    (OUT/'adaptive_gate_product_results.json').write_text(json.dumps(out,indent=2))
    plt.yscale('log');plt.xlabel('t');plt.ylabel('semi-discrete relative error');plt.legend(fontsize=8);plt.grid(alpha=.2);plt.tight_layout();plt.savefig(FIG/'adaptive_gate_product_test.pdf');plt.close()
