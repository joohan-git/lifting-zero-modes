"""Geometry / R_F only sufficient classifier. No lifted spectral input.

Frozen after retrospective development. Graph fractional coordinates, edge
translations, physical cell, contact direction stars and graph symmetry
operations are the complete inputs. The operations are verified locally.
Generic rank/sector tests sample 12 deterministic off-symmetry wavevectors;
they are numerical generic tests, not symbolic proofs for every q.
"""
import numpy as np
import scipy.linalg as la
import spglib

TOL = 1e-8
Q_SEED = 20261120
NQ = 12
CUBIC = {'23', 'm-3', '432', '-43m', 'm-3m'}
UNIAXIAL = {'3','-3','32','3m','-3m','4','-4','4/m','422','4mm','-42m','4/mmm',
            '6','-6','6/m','622','6mm','-6m2','6/mmm'}

def close(a, b, scale=None):
    a,b=np.asarray(a),np.asarray(b)
    den=max(la.norm(a),la.norm(b)) if scale is None else scale
    return bool(la.norm(a-b) <= TOL*den) if den else bool(la.norm(a-b)==0)

def compatibility(g,q):
    R=np.zeros((len(g['edges']),3*g['n']),complex)
    for b,(i,j,s) in enumerate(g['edges']):
        u=g['unit'][b]
        R[b,3*i:3*i+3]-=u
        R[b,3*j:3*j+3]+=np.exp(2j*np.pi*np.dot(s,q))*u
    return R

def kernel(R):
    _,s,v=la.svd(R,full_matrices=True,check_finite=False)
    rank=int(np.sum(s>TOL*s[0])) if len(s) and s[0] else 0
    return v[rank:].conj().T

def local_data(g, stars):
    bonds=[[] for _ in range(g['n'])]
    for b,(i,j,s) in enumerate(g['edges']):
        bonds[i].append(g['unit'][b]);bonds[j].append(-g['unit'][b])
    T=[];S=[];normals={}
    for i,(bb,ss) in enumerate(zip(bonds,stars)):
        B=np.asarray(bb).reshape(-1,3);C=np.asarray(ss).reshape(-1,3)
        T.append(B.T@B);S.append(C.T@C)
        _,sv,v=la.svd(B,full_matrices=True,check_finite=False)
        rank=int(np.sum(sv>TOL*sv[0])) if len(sv) and sv[0] else 0
        if rank==2:normals[i]=v[-1]
    return bonds,np.array(T),np.array(S),normals

def groups(g, stars, bonds):
    """Site groups of supplied graph operations that preserve BOTH local stars.
    Filters non-isometries introduced by a common-cell metric deformation.
    No scaffold or lifted spectrum is inspected.
    """
    A=g['A'];inv=la.inv(A.T);out=[];rots=[]
    def invariant(star,U):
        x=np.asarray(star).reshape(-1,3)
        if not len(x):return True
        distances=la.norm((x@U.T)[:,None,:]-x[None,:,:],axis=2)
        return bool(distances.min(1).max()<=TOL)
    for x,ss,bb in zip(g['pts'],stars,bonds):
        rr=[];cc=[]
        for r,t in g.get('symmetry_ops',[(np.eye(3),np.zeros(3))]):
            if la.norm((r@x+t-x+.5)%1-.5)>TOL:continue
            U=A.T@r@inv
            if close(U@U.T,np.eye(3)) and invariant(ss,U) and invariant(bb,U):
                rr.append(np.rint(r).astype(int));cc.append(U)
        answer=spglib.get_pointgroup(np.array(rr)) if rr else None
        out.append(answer[0].strip() if answer else 'unknown');rots.append(cc)
    return out,rots

def bipartition(g):
    adj=[[] for _ in range(g['n'])]
    for i,j,s in g['edges']:adj[i].append(j);adj[j].append(i)
    col=np.full(g['n'],-1,int)
    for root in range(g['n']):
        if col[root]>=0:continue
        col[root]=0;stack=[root]
        while stack:
            i=stack.pop()
            for j in adj[i]:
                if col[j]<0:col[j]=1-col[i];stack.append(j)
                elif col[j]==col[i]:return None
    return col

def classify(g,stars):
    if g['d']!=3:raise ValueError('Frozen protocol is three-dimensional')
    bonds,T,S,normals=local_data(g,stars)
    qs=np.random.default_rng(Q_SEED).random((NQ,3))
    kernels=[kernel(compatibility(g,q)) for q in qs]
    ms=[z.shape[1] for z in kernels]
    out=dict(prediction='abstain',route=None,values=[],M_F=sorted(set(ms)),
             planar_sites=sorted(normals),n_planar=len(normals),
             rank_q_seed=Q_SEED,rank_q_count=NQ,tolerance_relative=TOL,
             qualification='conditional on generic rigid scaffold and full positive lifting')
    def yes(route,values,**extra):
        out.update(prediction='predicted_flat',route=route,values=sorted(float(x) for x in values),**extra)
        return out
    if len(set(ms))!=1:out['reason']='generic_rank_not_constant';return out
    m=ms[0]
    if not m:out['reason']='no_generic_F_mechanisms';return out
    # L does NOT require S*n parallel to n and permits nonplanar spectator sites.
    if m==len(normals):
        vals=[n@S[i]@n for i,n in normals.items()]
        residual=max((la.norm(S[i]@n-(n@S[i]@n)*n)/max(la.norm(S[i]),1e-300)
                      for i,n in normals.items()),default=0.)
        return yes('L',vals,normal_not_S_eigenvector_max_relative=float(residual),
                   theorem='disjoint site-supported normals exhaust ker R_F; diagonal compression')
    hm,rots=groups(g,stars,bonds);out['contact_site_groups']=hm
    # U: a common scalar on the E spaces, and R_F alone confines its kernel
    # to those spaces. This deliberately abstains for unequal E eigenvalues.
    axes=[];ev=[];valid=all(h in UNIAXIAL for h in hm)
    if valid:
        for i in range(g['n']):
            axis=None
            for U in rots[i]:
                if la.det(U)<0:continue
                angle_cos=(np.trace(U)-1)/2
                if min(abs(angle_cos-x) for x in (-.5,0.,.5))<TOL:
                    _,_,vv=la.svd(U-np.eye(3));axis=vv[-1];break
            if axis is None:valid=False;break
            P=np.eye(3)-np.outer(axis,axis);val=np.trace(P@S[i])/2
            valid &= close(P@S[i]@P,val*P,la.norm(S[i]))
            axes.append(axis);ev.append(val)
        if valid:
            valid &= close(np.array(ev),np.full(len(ev),np.mean(ev)))
            err=max(la.norm(np.array([axes[i]@z[3*i:3*i+3] for i in range(g['n'])]))/
                    max(la.norm(z),1e-300) for z in kernels)
            valid &= err<=TOL
            if valid:return yes('U',[np.mean(ev)]*m,sector_leakage_relative=float(err))
    # C/P2. Isotropic bond/contact tensors constant within each colour.
    col=bipartition(g)
    if col is not None and all(h in CUBIC for h in hm):
        tt=np.trace(T,axis1=1,axis2=2)/3;ss=np.trace(S,axis1=1,axis2=2)/3
        valid=all(close(t,np.eye(3)*v) and close(s,np.eye(3)*w)
                  for t,v,s,w in zip(T,tt,S,ss))
        valid &= all(close(tt[col==k],np.full(sum(col==k),np.mean(tt[col==k]))) and
                     close(ss[col==k],np.full(sum(col==k),np.mean(ss[col==k]))) for k in (0,1))
        if valid:
            ta,tb=[np.mean(tt[col==k]) for k in (0,1)]
            sa,sb=[np.mean(ss[col==k]) for k in (0,1)]
            return yes('C',[(tb*sa+ta*sb)/(ta+tb)]*m)
    # Sector composition: local A normals + transverse bipartite P2;
    # periodic axial B chains forbid B axial displacement at generic q.
    aids=sorted(normals);bids=[i for i in range(g['n']) if i not in normals]
    if aids and bids:
        axis=normals[aids[0]];P=np.eye(3)-np.outer(axis,axis)
        valid=all(abs(abs(normals[i]@axis)-1)<=TOL for i in aids)
        classes=np.array([0 if i in normals else 1 for i in range(g['n'])])
        adj={i:[] for i in bids}
        for b,(i,j,sh) in enumerate(g['edges']):
            dot=abs(g['unit'][b]@axis)
            if classes[i]!=classes[j]:valid &= dot<=TOL
            elif classes[i]==1 and abs(dot-1)<=TOL:
                adj[i].append((j,np.array(sh)));adj[j].append((i,-np.array(sh)))
            else:valid=False
        seen=set()
        for root in bids:
            if root in seen:continue
            stack=[root];pot={root:np.zeros(3,int)};periodic=False;seen.add(root)
            while stack:
                i=stack.pop()
                for j,sh in adj[i]:
                    if j not in seen:seen.add(j);pot[j]=pot[i]+sh;stack.append(j)
                    elif np.any(pot[i]+sh-pot[j]):periodic=True
            valid &= periodic
        tv=[];sv=[]
        for ids in (aids,bids):
            tt=np.array([np.trace(P@T[i])/2 for i in ids]);ss=np.array([np.trace(P@S[i])/2 for i in ids])
            valid &= close(tt,np.full(len(tt),np.mean(tt))) and close(ss,np.full(len(ss),np.mean(ss)))
            for i,t,s in zip(ids,tt,ss):
                valid &= close(P@T[i]@P,t*P,la.norm(T[i])) and close(P@S[i]@P,s*P,la.norm(S[i]))
                valid &= la.norm(P@S[i]@axis)<=TOL*la.norm(S[i])
            tv.append(float(np.mean(tt)));sv.append(float(np.mean(ss)))
        if valid and m>=len(aids):
            val=(tv[1]*sv[0]+tv[0]*sv[1])/sum(tv)
            values=[axis@S[i]@axis for i in aids]+[val]*(m-len(aids))
            return yes('mixed_L_P2',values)
    out['reason']='no_sufficient_route';return out
