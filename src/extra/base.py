import os,sys,re; os.chdir('/home/claude/w/b'); sys.path.insert(0,'.')
import numpy as np
src=open('nb_src.py').read(); cut=src.index("# %% [markdown]\n# ## 3. Shared preprocessing")
ns={}; exec(compile(re.sub(r'^# %%.*$','',src[:cut],flags=re.M).replace("plt.show()","plt.close('all')"),'h','exec'),ns)
y,ydev,yte=ns['y'],ns['ydev'],ns['yte']
c=np.bincount(ydev,minlength=50); o=np.argsort(-c)
print('top1 baseline acc test',(yte==o[0]).mean(),'top3 freq baseline test',np.isin(yte,o[:3]).mean(),'top5',np.isin(yte,o[:5]).mean(), 'random top3',3/50)
