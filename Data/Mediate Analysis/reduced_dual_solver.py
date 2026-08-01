import numpy as np, itertools, time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

# ===========================================================================
#  Mediation partial identification, NIE(0), parallel (natural/manipulated) design.
#  - make_data : continuous 2-arm SCM, discretized to (kY,kM) bins; response
#                functions drawn directly so p0,p1 come from ONE q_true and the
#                full primal is feasible by construction.
#  - full_primal : the exact LP over all |M|^2 |Y|^{2|M|} response-function
#                  strata (the "full program").
#  - reduced_dual : the LP after the two-stage max-reduction, |M|^2 aggregated
#                   feasibility rows + 6|M| epigraph families (the corrected,
#                   per-pair form -- NO further m0/m1 split).
#  - verify_closed_form : brute-force max over strata == closed-form Psi(lambda),
#                   an independent check of the reduction algebra itself.
#  Estimand cost:  c = v_{y_{0,m1}} - v_{y_{0,m0}}  (NIE(0)).
# ===========================================================================

def make_data(kM, kY, N=200000, seed=0,
              aD=1.0, aU=1.2, sM=0.5, bD=0.6, bM=1.0, bU=0.9, sY=0.5):
    rng = np.random.default_rng(seed)
    U  = rng.standard_normal(N)
    eM = rng.standard_normal(N)*sM
    eY = rng.standard_normal(N)*sY
    mlat = np.stack([aD*d + aU*U + eM for d in (0,1)], axis=1)              # N x 2
    medges = np.quantile(mlat.ravel(), np.linspace(0,1,kM+1))
    medges[0], medges[-1] = -np.inf, np.inf
    mbin = np.stack([np.clip(np.digitize(mlat[:,d], medges[1:-1]),0,kM-1) for d in (0,1)],1)
    lab_all = np.clip(np.digitize(mlat.ravel(), medges[1:-1]),0,kM-1)
    midM = np.array([mlat.ravel()[lab_all==j].mean() if np.any(lab_all==j) else 0.0
                     for j in range(kM)])
    ylat = np.stack([[bD*d + bM*midM[j] + bU*U + eY for j in range(kM)] for d in (0,1)],0) # 2 x kM x N
    yedges = np.quantile(ylat.ravel(), np.linspace(0,1,kY+1)); yedges[0],yedges[-1]=-np.inf,np.inf
    ybin = np.clip(np.digitize(ylat, yedges[1:-1]),0,kY-1)                  # 2 x kM x N
    ylab = np.clip(np.digitize(ylat.ravel(), yedges[1:-1]),0,kY-1)
    vY = np.array([ylat.ravel()[ylab==a].mean() if np.any(ylab==a) else 0.0
                   for a in range(kY)])
    p0 = np.zeros((2,kM,kY)); p1 = np.zeros((2,kM,kY))
    for d in (0,1):
        md = mbin[:,d]
        yd = ybin[d, md, np.arange(N)]                                     # y_{d, m_d}
        for m in range(kM):
            sel = md==m
            for y in range(kY):
                p0[d,m,y] = np.mean(sel & (yd==y))
        for m in range(kM):
            for y in range(kY):
                p1[d,m,y] = np.mean(ybin[d,m,:]==y)
    y0m1 = ybin[0, mbin[:,1], np.arange(N)]
    y0m0 = ybin[0, mbin[:,0], np.arange(N)]
    true_nie0 = np.mean(vY[y0m1]-vY[y0m0])
    return p0, p1, vY, true_nie0

# ---------------------------------------------------------------------------
def full_primal(kM, kY, p0, p1, vY):
    Ms = list(itertools.product(range(kM), repeat=2))                      # (m0,m1)
    Ys = list(itertools.product(range(kY), repeat=2*kM))                   # y_{d,j}
    def yidx(d,j): return d*kM+j
    strata = [(m,y) for m in Ms for y in Ys]
    S = len(strata)
    rows0 = {(d,m,y):i for i,(d,m,y) in enumerate(itertools.product(range(2),range(kM),range(kY)))}
    off0  = len(rows0)
    rows1 = {(d,m,y):off0+i for i,(d,m,y) in enumerate(itertools.product(range(2),range(kM),range(kY)))}
    norm  = off0+len(rows1); R = norm+1
    A = lil_matrix((R,S)); c = np.zeros(S); b = np.zeros(R)
    for d in range(2):
        for m in range(kM):
            for y in range(kY):
                b[rows0[(d,m,y)]] = p0[d,m,y]; b[rows1[(d,m,y)]] = p1[d,m,y]
    b[norm]=1.0
    for si,(m,y) in enumerate(strata):
        m0,m1=m
        c[si]= vY[y[yidx(0,m1)]] - vY[y[yidx(0,m0)]]
        for d in (0,1):
            md=m[d]; yv=y[yidx(d,md)]; A[rows0[(d,md,yv)], si]+=1
        for d in (0,1):
            for j in range(kM):
                A[rows1[(d,j,y[yidx(d,j)])], si]+=1
        A[norm,si]=1.0
    A=csr_matrix(A); res={}
    for sense,lab in [(1,'lo'),(-1,'hi')]:
        r=linprog(sense*c, A_eq=A, b_eq=b, bounds=(0,None), method='highs')
        res[lab]= sense*r.fun if r.success else None
    return res, S

# ---------------------------------------------------------------------------
def reduced_dual(kM, kY, p0, p1, vY, upper=False):
    idx={}; k=0
    for e in (0,1):
        for d in (0,1):
            for j in range(kM):
                for a in range(kY): idx[('lam',e,d,j,a)]=k; k+=1
    idx[('void',)]=k; k+=1
    for fam in ['s1','s0','u1','u0','p','n']:
        for j in range(kM): idx[(fam,j)]=k; k+=1
    V=k; obj=np.zeros(V)
    for d in (0,1):
        for j in range(kM):
            for a in range(kY):
                obj[idx[('lam',0,d,j,a)]]=p0[d,j,a]; obj[idx[('lam',1,d,j,a)]]=p1[d,j,a]
    obj[idx[('void',)]]=1.0
    Arows=[]; brows=[]
    def add(coefs, rhs):
        row=np.zeros(V)
        for key,val in coefs: row[idx[key]]+=val
        Arows.append(row); brows.append(rhs)
    s=1.0 if not upper else -1.0
    for j in range(kM):
        for a in range(kY):
            add([(('s1',j),-s),(('lam',1,1,j,a),s)],0)                      # s1_j >= lam_{1,1,j,a}
            add([(('s0',j),-s),(('lam',1,0,j,a),s)],0)                      # s0_j >= lam_{1,0,j,a}
            add([(('u1',j),-s),(('lam',1,1,j,a),s),(('lam',0,1,j,a),s)],0)  # u1_j >= lam_{1,1}+lam_{0,1}
            add([(('u0',j),-s),(('lam',1,0,j,a),s),(('lam',0,0,j,a),s)],0)  # u0_j >= lam_{1,0}+lam_{0,0}
            add([(('p',j),-s),(('lam',1,0,j,a),s),(('lam',0,0,j,a),s)], (-s)*vY[a])  # p_j >= lam+lam+v
            add([(('n',j),-s),(('lam',1,0,j,a),s)], (s)*vY[a])              # n_j >= lam_{1,0} - v
    for m0 in range(kM):
        for m1 in range(kM):
            coefs=[(('void',),1.0)]
            if m0!=m1:
                coefs+=[(('u1',m1),1.0)]+[(('s1',j),1.0) for j in range(kM) if j!=m1]
                coefs+=[(('p',m0),1.0),(('n',m1),1.0)]+[(('s0',j),1.0) for j in range(kM) if j not in (m0,m1)]
            else:
                coefs+=[(('u1',m0),1.0)]+[(('s1',j),1.0) for j in range(kM) if j!=m0]
                coefs+=[(('u0',m0),1.0)]+[(('s0',j),1.0) for j in range(kM) if j!=m0]
            add(coefs,0) if not upper else add([(kk,-v) for kk,v in coefs],0)
    Aub=np.array(Arows); bub=np.array(brows)
    sense= -1.0 if not upper else 1.0
    r=linprog(sense*obj, A_ub=Aub, b_ub=bub, bounds=(None,None), method='highs')
    return (sense*r.fun) if r.success else None