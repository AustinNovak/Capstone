import os; os.chdir('/home/claude/w/b')
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=pd.read_csv('results/regional_models_test_results.csv'); r['a']=r.arm.str.split(':').str[0]
fs=pd.read_csv('results/regional_feature_sets.csv'); regs=list(fs.region)
feats=['LAT_DD','LON_DD','month','lcond','PHSTVL','ECOREGL3','YEAR','Temp','hardness','do','TP','LWSKM2','CA','SO4','NO3','CL','alk','mg','na','k','tn','SO4HCO3','ratio','rbpscore','Dataset','HUC','state']
H=np.zeros((len(feats),len(regs)))  # 2 selected,1 available not selected,0 absent/dropped
for j,rg in enumerate(regs):
    row=fs.iloc[j]; av=set(row.available.split(', ')); sel=set(row.region_best.split(', '))
    for i,f in enumerate(feats): H[i,j]=2 if f in sel else (1 if f in av else 0)
fig,ax=plt.subplots(1,3,figsize=(17,6.5),gridspec_kw={'width_ratios':[1.1,1,1]})
from matplotlib.colors import ListedColormap
ax[0].imshow(H,cmap=ListedColormap(['#e8e8e8','#9ecae1','#08519c']),aspect='auto')
ax[0].set_xticks(range(len(regs))); ax[0].set_xticklabels([x.replace('West Virginia','WV').replace('Pennsylvania','PA').replace('Kentucky','KY') for x in regs]); ax[0].set_yticks(range(len(feats))); ax[0].set_yticklabels(feats,fontsize=8)
ax[0].set_title('Feature availability by region\ngrey = unavailable/constant, light = available, dark = region-best')
for k,(m,t) in enumerate([('test_macro_f1','Macro-F1'),('test_top3','Top-3 accuracy')]):
    a=ax[k+1]; w=.38; x=np.arange(len(regs))
    g=[r[(r.region==rg)&(r.model=='Random Forest')&(r.a=='GC')][m].iloc[0] for rg in regs]
    b=[r[(r.region==rg)&(r.model=='Random Forest')&(r.a=='RB')][m].iloc[0] for rg in regs]
    a.bar(x-w/2,g,w,label='One global model (expanded)',color='#999'); a.bar(x+w/2,b,w,label='Region-specific model (region-best features)',color='#08519c')
    a.set_xticks(x); a.set_xticklabels([f'{rg.split()[0] if rg!="VA-MD-NJ" else rg}\n({int(n)} test sites)' for rg,n in zip(regs,fs.test_sites)],fontsize=8); a.set_title(f'{t}, held-out test sites (RF)'); a.set_ylabel(t)
    if k==0: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/regional_models.png',dpi=150)
print(r[(r.model=='Random Forest')&(r.a.isin(['GC','RB']))][['region','a','n_features','test_macro_f1','test_top3']].to_string())
