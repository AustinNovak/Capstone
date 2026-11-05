import os, sys, pickle, re
os.chdir('/home/claude/w/b'); sys.path.insert(0,'/home/claude/w/b')
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
R,F,rte=pickle.load(open('cache/region_results.pkl','rb'))
import pandas as pd
d=pd.read_csv('cleaned_dataset_expanded.csv',dtype={'SITE_ID':str})  # only for site ids / labels order
# rebuild test index + labels exactly as in the notebook
src=open('nb_src.py').read(); cut=src.index("# %% [markdown]\n# ## 3. Shared preprocessing")
code=re.sub(r'^# %%.*$','',src[:cut],flags=re.M).replace("plt.show()","plt.close('all')")
ns={}; exec(compile(code,'h','exec'),ns); yte=ns['yte']; gte=ns['gte']; K=ns['K']
def top3(full,y): return float((np.argsort(full,1)[:,-3:]==np.asarray(y)[:,None]).any(1).mean())
regs=['Pennsylvania','Ohio','Kentucky','West Virginia','VA-MD-NJ']; models=['Random Forest','Extra Trees (new)','Gradient Boosting']
arms=sorted(R.arm.unique()); short=lambda a:a.split(':')[0]
# assemble pooled predictions
POOL={}
for m in models:
    for a in arms:
        pred=np.zeros(len(yte),int); full=np.zeros((len(yte),K))
        for r in regs:
            row=R[(R.region==r)&(R.model==m)&(R.arm==a)].iloc[0]; idx=np.where(rte==r)[0]; pred[idx]=row._pred; full[idx]=row._full
        POOL[(m,a)]=(pred,full)
rows=[]
for (m,a),(p,f) in POOL.items():
    rows.append(dict(model=m,arm=a,macro_f1=f1_score(yte,p,average='macro',zero_division=0),accuracy=accuracy_score(yte,p),top3=top3(f,yte)))
P=pd.DataFrame(rows); P['arm']=P.arm.map(short); P=P.pivot(index='model',columns='arm',values=['macro_f1','accuracy','top3']).round(4); print(P.to_string()); P.to_csv('results/regional_pooled_test_results.csv')
# site-level bootstrap of differences
rs=np.random.RandomState(42); sites=np.unique(gte); by={s:np.where(gte==s)[0] for s in sites}; NB=200
def bs(m,a,b,mask):
    A=POOL[(m,[x for x in arms if short(x)==a][0])]; B=POOL[(m,[x for x in arms if short(x)==b][0])]
    ss=[s for s in sites if mask[by[s][0]]]; out=[]
    for _ in range(NB):
        idx=np.concatenate([by[s] for s in rs.choice(ss,len(ss))]); y=yte[idx]
        out.append([f1_score(y,B[0][idx],average='macro',zero_division=0)-f1_score(y,A[0][idx],average='macro',zero_division=0), top3(B[1][idx],y)-top3(A[1][idx],y)])
    o=np.array(out); return o.mean(0),np.percentile(o,2.5,0),np.percentile(o,97.5,0)
res=[]
for m in models:
    for scope,mask in [('All regions pooled',np.ones(len(yte),bool))]+[(r,rte==r) for r in regs]:
        for a,b,lab in [('GC','RB','region-best model vs one global expanded model'),('G7','R7','region model vs global model, both original 7')]:
            mu,lo,hi=bs(m,a,b,mask)
            for i,mt in enumerate(['macro_f1','top3']): res.append(dict(model=m,scope=scope,comparison=lab,metric=mt,diff=mu[i],ci_low=lo[i],ci_high=hi[i],excludes_zero=bool(lo[i]>0 or hi[i]<0)))
B=pd.DataFrame(res); B.round(5).to_csv('results/regional_vs_global_bootstrap.csv',index=False)
print(B[(B.model=='Random Forest')&(B.comparison.str.startswith('region-best'))].round(4).to_string())
print(B[(B.scope=='All regions pooled')].round(4).to_string())
