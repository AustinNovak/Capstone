import re, sys, os
os.chdir('/home/claude/w/b'); sys.path.insert(0,'/home/claude/w/b')
src=open('nb_src.py').read()
cut=src.index("# %% [markdown]\n# ## 5. Feature selection")
code=re.sub(r'^# %%.*$','',src[:cut],flags=re.M)
code=code.replace("plt.show()","plt.close('all')")   # no display; figures already saved before
ns={}
exec(compile(code,'nb_head','exec'),ns)
import numpy as np, pandas as pd, shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ORIG7=ns['ORIG7']; SHAP=ns['SHAP']; Xsh=ns['Xsh'].reset_index(drop=True); le=ns['le']; FINAL_B=ns['FINAL_B']; full_proba=ns['full_proba']; ydev=ns['ydev']
# numeric coding for colour (categoricals use an integer code, ordered by the share of rows - see caption)
num=Xsh.copy()
for c in ['state','ECOREGL3']:
    num[c]=pd.Categorical(Xsh[c]).codes.astype(float)
num=num.astype(float)
top=pd.Series(ydev).value_counts().index[:4]; gnames=[le.classes_[g] for g in top]
print('genera',gnames, 'states in sample',sorted(Xsh.state.unique()))
os.makedirs('figures',exist_ok=True)
def panel(ax,vals,title):
    plt.sca(ax)
    ex=shap.Explanation(values=vals,data=num.values,feature_names=ORIG7)
    shap.plots.beeswarm(ex,max_display=7,show=False,plot_size=None,color_bar=False)
    ax.set_title(title,fontsize=10,fontweight='bold'); ax.set_xlabel('SHAP value (effect on probability)',fontsize=8)
for n in list(ns['SPECS'])[:4]:
    S=SHAP[n]; model=FINAL_B[n]['est']; pred=full_proba(model,Xsh).argmax(1)
    pooled=S[np.arange(len(S)),:,pred]                       # SHAP of each row's predicted genus
    fig,axes=plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel(); fig.subplots_adjust(hspace=0.32,wspace=0.28)
    for ax,g,gn in zip(axes,top,gnames): panel(ax,S[:,:,g],f'Effect on P({gn})')
    sm=plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue); sm.set_array([0,1])
    cb=fig.colorbar(sm,ax=axes,fraction=0.012,pad=0.01,ticks=[0,1]); cb.ax.set_yticklabels(['Low','High']); cb.set_label('Feature value',fontsize=9)
    fig.suptitle(f'SHAP beeswarm - {n} (original 7 features)\neach dot = one row; right of 0 = raises that genus\'s probability; colour = feature value',fontweight='bold',y=0.99)
    fn=f"figures/shap_beeswarm_{n.split()[0].lower()}.png"; fig.savefig(fn,dpi=140,bbox_inches='tight'); plt.close(fig); print('saved',fn)
# expanded Random Forest (TreeExplainer, exact)
import pickle; FC=pickle.load(open('cache/final_C.pkl','rb')); EXP=ORIG7+pickle.load(open('cache/final_sel.pkl','rb'))['additions']; rfC=FC['Random Forest']['est']; pre=rfC.named_steps['pre']
rows=ns['Xdev'].iloc[ns['sh_idx']][EXP]; Z=pre.transform(rows); names=list(pre.get_feature_names_out())
tsh=shap.TreeExplainer(rfC.named_steps['model']).shap_values(Z,check_additivity=False); tsh=np.stack(tsh,-1) if isinstance(tsh,list) else tsh
# collapse encoded columns back to the 13 original columns (sum of SHAP; colour = raw numeric value)
def col_of(nm):
    b=nm.split('__',1)[1]
    for f in sorted(EXP,key=len,reverse=True):
        if b==f or b.startswith(f+'_') or b.startswith('missingindicator_'+f): return f
cols=[col_of(nm) for nm in names]
agg=np.stack([tsh[:,[i for i,c in enumerate(cols) if c==f],:].sum(1) for f in EXP],1)   # (n,13,classes)
numE=rows.reset_index(drop=True).copy()
for c in ['state','ECOREGL3','HUC']: numE[c]=pd.Categorical(numE[c]).codes.astype(float)
numE=numE.astype(float)
predE=full_proba(rfC,rows).argmax(1); pooledE=agg[np.arange(len(agg)),:,predE]
fig,axes=plt.subplots(2,2,figsize=(13,10)); axes=axes.ravel(); fig.subplots_adjust(hspace=0.3,wspace=0.3)
for ax,vals,t in zip(axes,[agg[:,:,g] for g in top],[f'Effect on P({g})' for g in gnames]):
    plt.sca(ax); shap.plots.beeswarm(shap.Explanation(values=vals,data=numE.values,feature_names=EXP),max_display=13,show=False,plot_size=None,color_bar=False)
    ax.set_title(t,fontsize=10,fontweight='bold'); ax.set_xlabel('SHAP value (effect on probability)',fontsize=8)
sm=plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue); sm.set_array([0,1]); cb=fig.colorbar(sm,ax=axes,fraction=0.012,pad=0.01,ticks=[0,1]); cb.ax.set_yticklabels(['Low','High']); cb.set_label('Feature value',fontsize=9)
fig.suptitle('SHAP beeswarm - Random Forest, expanded 13 features\n(grey dots = value missing and imputed; missing-flag columns folded into their feature)',fontweight='bold',y=0.99)
fig.savefig('figures/shap_beeswarm_expanded_rf.png',dpi=140,bbox_inches='tight'); print('saved expanded')
# numeric summary of directions for the report (Spearman of feature value vs SHAP, predicted-genus pooled), original 7
from scipy.stats import spearmanr
rows_=[]
for n in list(ns['SPECS'])[:4]:
    S=SHAP[n]; pred=full_proba(FINAL_B[n]['est'],Xsh).argmax(1); pooled=S[np.arange(len(S)),:,pred]
    for j,f in enumerate(ORIG7): rows_.append(dict(model=n,feature=f,spearman=spearmanr(num[f],pooled[:,j])[0]))
pd.DataFrame(rows_).pivot(index='feature',columns='model',values='spearman').round(2).to_csv('results/shap_direction_spearman_original7.csv')
print(pd.read_csv('results/shap_direction_spearman_original7.csv'))

# direction table per genus (Spearman: feature value vs SHAP), numeric features only
from scipy.stats import spearmanr
out=[]
for n in list(ns['SPECS'])[:4]:
    S=SHAP[n]
    for g,gn in zip(top,gnames):
        for j,f in enumerate(ORIG7):
            if f in ('state','ECOREGL3'): continue
            out.append(dict(model=n,genus=gn,feature=f,spearman=round(spearmanr(num[f],S[:,j,g])[0],2)))
D=pd.DataFrame(out); D.to_csv('results/shap_direction_by_genus_original7.csv',index=False)
print(D[D.model=='Random Forest'].pivot(index='feature',columns='genus',values='spearman'))
outE=[]
for g,gn in zip(top,gnames):
    for j,f in enumerate(EXP):
        if f in ('state','ECOREGL3','HUC'): continue
        m=~rows.reset_index(drop=True)[f].isna().values
        outE.append(dict(genus=gn,feature=f,spearman=round(spearmanr(numE[f][m],agg[m,j,g])[0],2)))
pd.DataFrame(outE).to_csv('results/shap_direction_by_genus_expanded_rf.csv',index=False)
print(pd.DataFrame(outE).pivot(index='feature',columns='genus',values='spearman'))
