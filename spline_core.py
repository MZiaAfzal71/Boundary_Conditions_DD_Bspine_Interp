"""Cubic spline constructions used by the article. No plotting or file I/O.

All public fit functions use normalized parameters on [0, 1], float64 arrays,
and explicit boundary conditions. Closed data repeat the first point once.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import BSpline, CubicSpline, PPoly
from scipy.spatial import cKDTree
from scipy.integrate import quad
from functools import lru_cache
import warnings


@dataclass
class Fit:
    curve: object
    t: np.ndarray
    knots: np.ndarray
    matrix: np.ndarray
    controls: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __call__(self, t, nu=0):
        return self.curve(t, nu=nu)


def validate(points, closed=False):
    d = np.asarray(points, dtype=float)
    if d.ndim != 2 or d.shape[1] != 2 or len(d) < 4:
        raise ValueError('Provide at least four ordered planar points.')
    if not np.isfinite(d).all():
        raise ValueError('All coordinates must be finite.')
    scale = float(np.linalg.norm(np.ptp(d, axis=0)))
    if scale == 0 or np.min(np.linalg.norm(np.diff(d, axis=0), axis=1)) < 1e-12*scale:
        raise ValueError('Consecutive duplicate or near-duplicate points; no silent removal.')
    if closed and np.linalg.norm(d[-1]-d[0]) > 1e-12*scale:
        raise ValueError('Closed data must repeat the first point exactly once.')
    if closed and len(d) < 5:
        raise ValueError('Use at least four distinct vertices for the periodic construction.')
    return d, scale


def parameters(points, alpha=0.5, closed=False):
    d, _ = validate(points, closed)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative.')
    h = np.linalg.norm(np.diff(d, axis=0), axis=1)**alpha
    t=np.r_[0.0, np.cumsum(h)/h.sum()]
    t[-1]=1.0
    return t


def extended_knots(s, closed=False, exterior='constant', exterior_scale=1.0):
    s = np.asarray(s, dtype=float)
    h = np.diff(s)
    if len(s) < 4 or np.min(h) <= 0 or exterior_scale <= 0:
        raise ValueError('Strictly increasing sites and positive exterior_scale required.')
    if closed:
        return np.r_[s[-4:-1]-(s[-1]-s[0]), s, s[1:4]+(s[-1]-s[0])]
    if exterior == 'constant':
        return np.r_[s[0]-exterior_scale*h[0]*np.arange(3,0,-1),
                     s, s[-1]+exterior_scale*h[-1]*np.arange(1,4)]
    if exterior == 'reflected':
        return np.r_[s[0]-exterior_scale*(s[1:4]-s[0])[::-1],
                     s, s[-1]+exterior_scale*(s[-1]-s[-4:-1])[::-1]]
    raise ValueError('exterior must be constant or reflected.')


def mapping(n, closed):
    """n intervals; n periodic unknowns or n+1 open unknowns; n+3 rows."""
    indices = np.arange(n+3) % n if closed else np.r_[0,np.arange(n+1),n]
    return np.eye(n if closed else n+1)[indices]


def coupled(points, alpha=0.5, closed=False, exterior='constant',
            exterior_scale=1.0, sites=None, breakpoints=None, frozen_exterior=None):
    d, _ = validate(points, closed)
    t = parameters(d, alpha, closed) if sites is None else np.asarray(sites)
    z = t.copy() if breakpoints is None else np.asarray(breakpoints)
    u = extended_knots(z, closed, exterior, exterior_scale)
    if frozen_exterior is not None and not closed:
        u[:3], u[-3:] = frozen_exterior[:3], frozen_exterior[-3:]
    if np.min(np.diff(u)) <= 0:
        raise ValueError('Knots must be strictly increasing.')
    n = len(d)-1
    r = mapping(n, closed)
    x = t[:-1] if closed else t
    b = BSpline.design_matrix(x,u,3,extrapolate=False).toarray()
    m = b@r
    p = np.linalg.solve(m,d[:-1] if closed else d)
    c = r@p
    return Fit(BSpline(u,c,3,extrapolate=False),t,u,m,c,
               {'method':'periodic' if closed else 'repeated', 'alpha':alpha,
                'closed':closed,'exterior':exterior,'exterior_scale':exterior_scale})


def averaged(points, alpha=0.5):
    d, _ = validate(points)
    t = parameters(d,alpha)
    n = len(d)-1
    interior = [np.mean(t[j:j+3]) for j in range(1,n-2)]
    u = np.r_[np.zeros(4),interior,np.ones(4)]
    m = BSpline.design_matrix(t,u,3,extrapolate=False).toarray()
    c = np.linalg.solve(m,d)
    return Fit(BSpline(u,c,3,extrapolate=False),t,u,m,c,
               {'method':'averaged','alpha':alpha,'closed':False})


def robin_reference(points, t, rleft, rright):
    """Independent piecewise-polynomial solve in second derivatives.

    C''(0)=rleft*C'(0), C''(1)=-rright*C'(1).
    rleft=rright=0 gives the classical natural cubic spline.
    """
    d = np.asarray(points)
    h = np.diff(t)
    slopes = np.diff(d,axis=0)/h[:,None]
    n = len(d)
    a = np.zeros((n,n)); rhs = np.zeros_like(d)
    a[0,0], a[0,1] = 1+rleft*h[0]/3, rleft*h[0]/6
    a[-1,-2], a[-1,-1] = rright*h[-1]/6,1+rright*h[-1]/3
    rhs[0],rhs[-1] = rleft*slopes[0],-rright*slopes[-1]
    for i in range(1,n-1):
        a[i,i-1:i+2] = h[i-1],2*(h[i-1]+h[i]),h[i]
        rhs[i] = 6*(slopes[i]-slopes[i-1])
    m = np.linalg.solve(a,rhs)
    coef = np.stack([(m[1:]-m[:-1])/(6*h[:,None]),m[:-1]/2,
                     slopes-h[:,None]*(2*m[:-1]+m[1:])/6,d[:-1]])
    return PPoly(coef,t,extrapolate=False)


def natural(points, alpha=0.5):
    """Natural second-derivative conditions in the same n+3 basis space."""
    d, _ = validate(points)
    t = parameters(d,alpha)
    u = extended_knots(t)
    b = BSpline(u,np.eye(len(d)+2),3,extrapolate=False)
    m = np.vstack([b(t),b(0,nu=2),b(1,nu=2)])
    c = np.linalg.solve(m,np.vstack([d,np.zeros((2,2))]))
    return Fit(BSpline(u,c,3,extrapolate=False),t,u,m,c,
               {'method':'natural','alpha':alpha,'closed':False})


def endpoint_approximation(points, alpha=0.5, support=None):
    """Open endpoint interpolation using interior points as coefficients.

    support: None, 'geometric', or 'speed_matched'. For support modes input
    must be closed; replace the first and last interior coefficients by
    D0 + ell_left*v and D0 - ell_right*v. Original parameter sites are kept.
    This is a fully specified study variant, not claimed as exact legacy code.
    """
    d, scale = validate(points,closed=support is not None)
    t=parameters(d,alpha,closed=support is not None)
    u=extended_knots(t)
    n=len(d)-1
    b=BSpline(u,np.eye(n+3),3,extrapolate=False)
    interior=d[1:-1].copy()
    meta={'method':support or 'endpoint','alpha':alpha,'closed':support is not None}
    if support is not None:
        v=d[1]-d[-2]
        if np.linalg.norm(v)<1e-12*scale:
            raise ValueError('Undefined seam direction: D1 and D[n-1] coincide.')
        v/=np.linalg.norm(v)
        ell_l=np.linalg.norm(d[1]-d[0])/2
        ell_r=np.linalg.norm(d[-1]-d[-2])/2
        lam_l=b(0,nu=1)[2]/(1-b(0)[2])
        lam_r=-b(1,nu=1)[n]/(1-b(1)[n])
        if support=='speed_matched': ell_r=lam_l*ell_l/lam_r
        elif support!='geometric': raise ValueError('Unknown support mode.')
        interior[0],interior[-1]=d[0]+ell_l*v,d[0]-ell_r*v
        meta.update(ell_left=ell_l,ell_right=ell_r,lambda_left=lam_l,lambda_right=lam_r)
    bl,br=b(0),b(1)
    pl=(d[0]-bl[2]*interior[0])/(bl[0]+bl[1])
    pr=(d[-1]-br[n]*interior[-1])/(br[n+1]+br[n+2])
    c=np.vstack([pl,pl,interior,pr,pr])
    return Fit(BSpline(u,c,3,extrapolate=False),t,u,np.eye(2),c,meta)


@lru_cache(maxsize=8)
def gauss_rule(order):
    return np.polynomial.legendre.leggauss(order)


def gauss_nodes(knots, order=48):
    z=np.unique(np.asarray(knots)[(np.asarray(knots)>=0)&(np.asarray(knots)<=1)])
    g,w=gauss_rule(order)
    mid=(z[:-1]+z[1:])/2; half=np.diff(z)/2
    return (mid[:,None]+half[:,None]*g).ravel(),(half[:,None]*w).ravel()


def energy(fit,scale=1.0,order=48):
    t,w=gauss_nodes(fit.knots,order)
    v,a=fit(t,1),fit(t,2)
    speed=np.linalg.norm(v,axis=1)
    if speed.min() < 1e-10*scale:
        return float('inf')
    cross=v[:,0]*a[:,1]-v[:,1]*a[:,0]
    return float(scale*np.sum(w*cross**2/speed**5))


def speed_stationary_points(fit):
    """All real stationary candidates of squared speed on cubic spans.

    Squared speed is quartic, so its derivative is cubic. Roots are numerical,
    not interval-certified, but this avoids relying only on a sampling grid.
    """
    pp=[PPoly.from_spline((fit.knots,fit.controls[:,j],3)) for j in range(2)]
    values=[]
    for i,(a,b) in enumerate(zip(pp[0].x[:-1],pp[0].x[1:])):
        if a<0 or b>1 or b<=a:continue
        vx=np.polyder(pp[0].c[:,i]);vy=np.polyder(pp[1].c[:,i])
        sq=np.polyadd(np.polymul(vx,vx),np.polymul(vy,vy))
        roots=np.roots(np.trim_zeros(np.polyder(sq),'f'))
        values.extend([a,b])
        values.extend(a+z.real for z in roots if abs(z.imag)<1e-9 and 0<z.real<b-a)
    return np.unique(values)


def bending_report(fit,scale):
    low,high=energy(fit,scale,48),energy(fit,scale,96)
    disagreement=abs(high-low)/max(abs(high),1e-14)
    stationary=speed_stationary_points(fit)
    minimum=float(np.min(np.linalg.norm(fit(stationary,1),axis=1))/scale)
    if minimum<1e-10:
        return float('inf'),float('inf'),disagreement,minimum,'nonregular'
    if disagreement<=1e-6 and minimum>=.01:
        return high,disagreement,disagreement,minimum,'gauss96'
    def integrand(t):
        v=fit(t,1);a=fit(t,2);s=np.linalg.norm(v)
        return scale*(v[0]*a[1]-v[1]*a[0])**2/s**5
    total=0.;err=0.;warning=False
    z=np.unique(fit.knots[(fit.knots>=0)&(fit.knots<=1)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for a,b in zip(z[:-1],z[1:]):
            margin=1e-10*(b-a)
            split=stationary[(stationary>a+margin)&(stationary<b-margin)]
            val,e=quad(integrand,a,b,epsabs=1e-8,epsrel=1e-8,limit=300,points=split)
            total+=val;err+=e
        warning=bool(caught)
    return total,err/max(abs(total),1e-14),disagreement,minimum,'adaptive_warning' if warning else 'adaptive'


def sampled_curve(fit,count=2049):
    # Include nodes in every span as well as the common uniform grid.
    z=np.unique(fit.knots[(fit.knots>=0)&(fit.knots<=1)])
    extra=np.concatenate([np.linspace(a,b,33) for a,b in zip(z[:-1],z[1:])])
    t=np.unique(np.r_[np.linspace(0,1,count),extra])
    return t,fit(t)


def seam_metrics(fit,scale):
    v=fit(np.array([0.,1.]),1); a=fit(np.array([0.,1.]),2)
    speed=np.linalg.norm(v,axis=1)
    den=max(float(speed.max()),1e-14*scale)
    cos=np.clip(np.dot(v[0],v[1])/max(float(np.prod(speed)),1e-28*scale**2),-1,1)
    curv=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(speed,1e-14*scale)**3
    return {'seam_position':float(np.linalg.norm(fit(0)-fit(1))/scale),
            'seam_angle_deg':float(np.degrees(np.arccos(cos))),
            'seam_d1':float(np.linalg.norm(v[0]-v[1])/den),
            'seam_d2':float(np.linalg.norm(a[0]-a[1])/max(float(np.linalg.norm(a,axis=1).max()),1e-14*scale)),
            'curvature_left':float(scale*curv[0]),'curvature_right':float(scale*curv[1]),
            'speed_left':float(speed[0]/scale),'speed_right':float(speed[1]/scale)}


def metrics(fit,points,truth=None,sample_count=2049):
    d,scale=validate(points)
    t,c=sampled_curve(fit,sample_count)
    bending,qerror,gdiff,minspeed,integrator=bending_report(fit,scale)
    out={'residual':float(np.max(np.linalg.norm(fit(fit.t)-d,axis=1))/scale),
         'condition':float(np.linalg.cond(fit.matrix)),
         'bending':bending,'quadrature_relative':qerror,'gauss_disagreement':gdiff,
         'integrator':integrator,'min_speed':minspeed,
         'min_span':float(np.min(np.diff(np.unique(fit.knots)))),
         'curve_sample_count':len(t)}
    if truth is not None:
        truth=np.asarray(truth)
        fwd=cKDTree(truth).query(c)[0]; back=cKDTree(c).query(truth)[0]
        out.update(hausdorff_sampled=float(max(fwd.max(),back.max())/scale),
                   rms_sampled=float(np.sqrt((np.mean(fwd*fwd)+np.mean(back*back))/2)/scale),
                   reference_sample_count=len(truth))
    if fit.metadata.get('closed'):out.update(seam_metrics(fit,scale))
    return out


def refine(points,closed=False,alpha=0.5,rho=0.25,condition_limit=1e8):
    """Exploratory deterministic coordinate search, fixed data parameters.

    |k_i| <= rho*min(h[i-1],h[i]) implies positive spans for rho<1/2.
    Objective: normalized bending + small baseline-departure and displacement
    penalties. No reference/ground-truth curve enters optimization.
    """
    if not 0<rho<.5:raise ValueError('rho must lie strictly between 0 and 0.5.')
    d,scale=validate(points,closed)
    base=coupled(d,alpha,closed); t=base.t; h=np.diff(t)
    bounds=rho*np.minimum(h[:-1],h[1:]); k=np.zeros(len(bounds))
    evals=0; rejects=0
    e0=energy(base,scale,24)
    sample=np.linspace(0,1,257); c0=base(sample)
    if not np.isfinite(e0) or e0<1e-12:
        return base,{'accepted':False,'reason':'degenerate baseline energy','evaluations':0}
    def candidate(k):
        nonlocal evals,rejects
        evals+=1
        try:
            z=t.copy();z[1:-1]+=k
            f=coupled(d,alpha,closed,sites=t,breakpoints=z,frozen_exterior=base.knots)
            if np.linalg.cond(f.matrix)>condition_limit:raise ValueError('conditioning')
            if np.max(np.linalg.norm(f(t)-d,axis=1))/scale>1e-9:raise ValueError('residual')
            e=energy(f,scale,24)
            j=e/e0+.1*np.mean(np.sum((f(sample)-c0)**2,axis=1))/scale**2+.01*np.mean((k/bounds)**2)
            if not np.isfinite(j):raise ValueError('nonregular quadrature')
            return float(j),f
        except (ValueError,np.linalg.LinAlgError):
            rejects+=1
            return float('inf'),None
    best,fit=candidate(k)
    for fraction in (1.,.5,.25):
        for i in range(len(k)):
            for sign in (-1,1):
                trial=k.copy();trial[i]=np.clip(k[i]+sign*fraction*bounds[i],-bounds[i],bounds[i])
                val,f=candidate(trial)
                if val<best-1e-10:k,best,fit=trial,val,f
    # Acceptance uses an independent, finer quadrature, not the search estimate.
    final_energy=energy(fit,scale,96); initial_energy=energy(base,scale,96)
    discrepancy=abs(final_energy-energy(fit,scale,48))/max(final_energy,1e-14)
    accepted=bool(final_energy<=initial_energy*(1+1e-8) and discrepancy<=1e-3)
    if not accepted:fit=base
    return fit,{'accepted':accepted,'objective24':best,'evaluations':evals,'rejected_candidates':rejects,
                'displacement':k.tolist(),'rho':rho,'final_quadrature_relative':discrepancy}
