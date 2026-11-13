import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=pd.read_csv('/home/claude/w/pkg/Milestone4_Revised/cleaned_dataset_expanded.csv',dtype={'SITE_ID':str})
g=d.GENUS.value_counts(); s=d.state.value_counts()
print(g.head(10).to_dict(), g.iloc[-1], len(g), d.SITE_ID.nunique())
fig,ax=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[2,1]})
t=g.head(20); ax[0].bar(range(20),t.values,color='#08519c'); ax[0].set_xticks(range(20)); ax[0].set_xticklabels(t.index,rotation=70,ha='right',fontsize=8); ax[0].set_yscale('log'); ax[0].set_ylabel('Fish records (log scale)'); ax[0].set_title(f'20 most common of {len(g)} genera (rarest has {g.iloc[-1]} rows)')
ax[1].bar(s.index,s.values,color='#6baed6'); ax[1].set_title('Records by state'); ax[1].set_ylabel('Fish records')
plt.tight_layout(); plt.savefig('eda.png',dpi=170)
mc=d[['LAT_DD','LON_DD','month','lcond','PHSTVL','LWSKM2','Temp','YEAR']].describe().T[['count','mean','50%','min','max']]; print(mc.round(2))
