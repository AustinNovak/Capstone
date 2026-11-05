import re, sys, os, pickle, time, json
os.chdir('/home/claude/w/b'); sys.path.insert(0,'/home/claude/w/b')
src=open('nb_src.py').read()
cut=src.index("# %% [markdown]\n# ## 6. Nested evaluation")
code=re.sub(r'^# %%.*$','',src[:cut],flags=re.M).replace("plt.show()","plt.close('all')")
ns={}; exec(compile(code,'nb_head','exec'),ns)
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
Xdev,ydev,gdev,Xte,yte,gte=[ns[k] for k in ['Xdev','ydev','gdev','Xte','yte','gte']]
CANDS=ns['CANDIDATES']; ORIG7=ns['ORIG7']; SPECS=ns['SPECS']; K=ns['K']; le=ns['le']
FB=ns['FINAL_B']; FC=pickle.load(open('cache/final_C.pkl','rb')); EXP=ORIG7+pickle.load(open('cache/final_sel.pkl','rb'))['additions']
REG={'PA':'Pennsylvania','OH':'Ohio','KY':'Kentucky','WV':'West Virginia','VA':'VA-MD-NJ','MD':'VA-MD-NJ','NJ':'VA-MD-NJ'}
rdev=Xdev.state.map(REG).values; rte=Xte.state.map(REG).values
SPECS['Gradient Boosting']['est'].set_params(early_stopping=False)   # regional subsets contain classes with 1 row, which breaks the stratified early-stopping split
MODELS=['Random Forest','Extra Trees (new)','Gradient Boosting']
os.makedirs('cache',exist_ok=True)
def cached(n,f):
    p=f'cache/{n}.pkl'
    if os.path.exists(p): return pickle.load(open(p,'rb'))
    o=f(); pickle.dump(o,open(p,'wb')); return o
def best_params(gs): return {k.replace('model__',''):(list(v) if isinstance(v,tuple) else v) for k,v in gs.best_params_.items()}
out=[]; featrows=[]
for reg in ['Pennsylvania','Ohio','Kentucky','West Virginia','VA-MD-NJ']:
    t0=time.time(); md=rdev==reg; mt=rte==reg
    Xr,yr,gr=Xdev[md].reset_index(drop=True),ydev[md],gdev[md]; Xt,yt=Xte[mt].reset_index(drop=True),yte[mt]
    nonmiss=(1-Xr[CANDS].isna().mean()); const=[c for c in CANDS if Xr[c].nunique(dropna=True)<=1]
    avail=[c for c in CANDS if nonmiss[c]>=0.60 and c not in const]
    r7=[c for c in ORIG7 if c in avail]
    print(f'== {reg}: dev {len(Xr):,} rows/{len(set(gr))} sites, test {len(Xt):,} rows/{len(set(gte[mt]))} sites; available {len(avail)}: {avail}; constant dropped: {const}',flush=True)
    def sel():
        ns['CANDIDATES']=avail; 
        try: return ns['run_selection'](Xr,yr,gr,tag=reg)
        finally: ns['CANDIDATES']=CANDS
    S=cached(f'reg_sel_{reg}',sel); adds=[a for a in S['additions'] if a not in r7]; rb=r7+adds
    featrows.append(dict(region=reg,dev_rows=len(Xr),dev_sites=len(set(gr)),test_rows=len(Xt),test_sites=len(set(gte[mt])),available=', '.join(avail),constant_dropped=', '.join(const),original7_used=', '.join(r7),additions=', '.join(adds) if adds else 'none',region_best=', '.join(rb)))
    print('   region-best features:',rb,flush=True)
    def fitall():
        o={}
        for n in MODELS:
            for arm,fs in (('R7: region data, original 7',r7),('RB: region data, region-best features',rb)):
                gs=ns['tune'](n,fs,Xr,yr,gr); o[(n,arm)]=dict(est=gs.best_estimator_,feats=fs,params=best_params(gs),cv=gs.best_score_)
        return o
    M=cached(f'reg_models_{reg}',fitall)
    def score(model,fs,Xq,yq):
        d,pred,full=ns['evaluate'](model,Xq[fs],yq); return d,pred,full
    for (n,arm),v in M.items():
        d,pred,full=score(v['est'],v['feats'],Xt,yt); out.append(dict(region=reg,model=n,arm=arm,n_features=len(v['feats']),selected_hyperparameters=str(v['params']),best_gridsearch_cv_macro_f1=v['cv'],test_macro_f1=d['macro_f1'],test_accuracy=d['accuracy'],test_top3=d['top3'],_pred=pred,_full=full))
    for n in MODELS:
        for arm,m,fs in (('G7: one global model, original 7',FB[n]['est'],ORIG7),('GC: one global model, expanded 13',FC[n]['est'],EXP)):
            d,pred,full=score(m,fs,Xt,yt); out.append(dict(region=reg,model=n,arm=arm,n_features=len(fs),selected_hyperparameters='(global model, trained on all development sites)',best_gridsearch_cv_macro_f1=np.nan,test_macro_f1=d['macro_f1'],test_accuracy=d['accuracy'],test_top3=d['top3'],_pred=pred,_full=full))
    print(f'   done {reg} in {(time.time()-t0)/60:.1f} min',flush=True)
R=pd.DataFrame(out); pickle.dump((R,pd.DataFrame(featrows),rte),open('cache/region_results.pkl','wb'))
R.drop(columns=['_pred','_full']).round(5).to_csv('results/regional_models_test_results.csv',index=False); pd.DataFrame(featrows).to_csv('results/regional_feature_sets.csv',index=False)
print('SAVED')
