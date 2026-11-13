import os,sys,re,pickle; os.chdir('/home/claude/w/b'); sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
src=open('nb_src.py').read(); cut=src.index("# %% [markdown]\n# ## 3. Shared preprocessing")
ns={}; exec(compile(re.sub(r'^# %%.*$','',src[:cut],flags=re.M).replace("plt.show()","plt.close('all')"),'h','exec'),ns)
Xte,yte,le,y=ns['Xte'],ns['yte'],ns['le'],ns['y']
C=pickle.load(open('cache/final_C.pkl','rb')); S=pickle.load(open('cache/final_sel.pkl','rb'))
print(C['Random Forest']['est'], list(Xte.columns)[:30])
est=C['Random Forest']['est']; p=est.predict(Xte); 
from sklearn.metrics import f1_score
print('macroF1',f1_score(yte,p,average='macro'))
top=[le.transform([g])[0] for g in ['Etheostoma','Rhinichthys','Lepomis','Semotilus','Notropis','Catostomus','Salmo','Micropterus','Salvelinus','Cottus']]
labs=[le.classes_[i] for i in top]
cm=confusion_matrix(yte,p,labels=top).astype(float); n=cm.sum(1,keepdims=True); cmn=cm/np.maximum(n,1)
fig,ax=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.1,1]})
im=ax[0].imshow(cmn,cmap='Blues',vmin=0,vmax=.6); ax[0].set_xticks(range(10)); ax[0].set_yticks(range(10)); ax[0].set_xticklabels(labs,rotation=60,ha='right',fontsize=8); ax[0].set_yticklabels(labs,fontsize=8)
for i in range(10):
    for j in range(10): ax[0].text(j,i,f'{cmn[i,j]:.2f}',ha='center',va='center',fontsize=6,color='w' if cmn[i,j]>.35 else 'k')
ax[0].set_xlabel('Predicted genus'); ax[0].set_ylabel('True genus'); ax[0].set_title('Test confusion matrix, 10 most common genera\n(row-normalised: share of each true genus predicted as each column)',fontsize=9)
rec=recall_score(yte,p,average=None,labels=range(len(le.classes_)),zero_division=0); cnt=np.bincount(yte,minlength=len(le.classes_))
o=np.argsort(-cnt)[:15]
ax[1].barh([le.classes_[i] for i in o][::-1],rec[o][::-1],color='#08519c'); ax[1].set_xlabel('Recall on held-out test rows'); ax[1].set_title('Recall for the 15 most common genera',fontsize=9)
plt.tight_layout(); plt.savefig('figures/final_confusion_recall.png',dpi=160)
print({le.classes_[i]:round(rec[i],2) for i in o}); print('genera with recall>0:',(rec>0).sum())
