"""D 0122: rolling corr window(60/120/250)×estimator + PCA/cluster leave-one-ETF-out. 결정적(난수 없음). raw → csv."""
import pandas as pd, numpy as np, warnings, sys, os; warnings.filterwarnings('ignore')
RAW=os.environ.get('ETF_RAW_DIR','data/raw')
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ari
OUT=sys.argv[1] if len(sys.argv)>1 else 'analysis/robustness/d0122'; os.makedirs(OUT,exist_ok=True)
U=pd.read_csv('config/universe.csv',dtype=str); TK=U.ticker.tolist(); SC=dict(zip(U.ticker,U.asset_scope))
C=pd.concat({t:pd.read_csv(f'{RAW}/{t}.csv',parse_dates=['Date']).set_index('Date').Close for t in TK},axis=1).sort_index()
R=np.log(C/C.shift(1)).iloc[1:]; Rt=R.copy(); Rt.loc[pd.to_datetime(['2019-03-14','2019-03-15']),'261240']=np.nan
DED=[t for t in TK if t not in ('102110','122630','114800')]; EQ=[t for t in DED if SC[t]=='domestic_equity']
ANC=['2020-03-19','2024-08-05','2025-04-07','2026-03-04']
# ---- 1. rolling avg pairwise corr
def roll(cols,w,kind):
    X=Rt[cols]; iu=np.triu_indices(len(cols),1); m=X['069500'].abs(); o=[]
    for i in range(w-1,len(X)):
        win=X.iloc[i-w+1:i+1]
        if kind=='spearman': win=win.rank()
        if kind=='pearson_ex_top2': win=win.drop(m.iloc[i-w+1:i+1].nlargest(2).index)
        o.append(np.nanmean(win.corr().values[iu]))
    return pd.Series(o,index=X.index[w-1:])
series={}; summ=[]
vol=R['069500'].rolling(20).std()
for w in [60,120,250]:
    for kind in ['pearson','spearman','pearson_ex_top2']:
        for grp,cols in [('domeq9',EQ),('dedup17',DED)]:
            s=roll(cols,w,kind); series[f'{grp}_{kind}_w{w}']=s
            d=s.diff().abs(); row=dict(group=grp,window=w,estimator=kind,mean=s.mean(),p99_abs_daily_change=d.quantile(.99),max_abs_daily_change=d.max(),
                spearman_vs_vol20=pd.concat([s,vol.rolling(w).mean()],axis=1).dropna().corr('spearman').iloc[0,1])
            for a in ANC:
                i=s.index.get_loc(pd.Timestamp(a)); row[f'jump_{a}']=s.iloc[i]-s.iloc[i-1]
            for y in [2020,2023,2024,2025,2026]: row[f'mean_{y}']=s[s.index.year==y].mean()
            summ.append(row)
pd.DataFrame(series).to_csv(f'{OUT}/rolling_corr_window_estimator_series.csv',float_format='%.6g')
pd.DataFrame(summ).to_csv(f'{OUT}/rolling_corr_window_estimator_summary.csv',index=False,float_format='%.6g')
# ---- 2. PCA LOETF (corr-based, = A 정의)
def pc(cols,X=None):
    X=(Rt[cols] if X is None else X).dropna(); w=np.linalg.eigvalsh(np.corrcoef(X.values.T))[::-1]; return w[0]/w.sum(), w[1]/w.sum()
rows=[]
for base,cols in [('all20',TK),('dedup17',DED)]:
    b1,b2=pc(cols); rows.append(dict(base=base,dropped='(none)',pc1=b1,pc2=b2,d_pc1=0.0))
    for t in cols:
        p1,p2=pc([c for c in cols if c!=t]); rows.append(dict(base=base,dropped=t,scope=SC[t],pc1=p1,pc2=p2,d_pc1=p1-b1))
    for nm,f in [('spearman_rank',lambda X:X.rank()),('cov_unscaled',None)]:
        X=Rt[cols].dropna()
        w=np.linalg.eigvalsh(X.rank().corr().values if nm=='spearman_rank' else X.cov().values)[::-1]; rows.append(dict(base=base,dropped=f'scaler:{nm}',pc1=w[0]/w.sum(),pc2=w[1]/w.sum(),d_pc1=w[0]/w.sum()-b1))
pd.DataFrame(rows).to_csv(f'{OUT}/pca_leave_one_etf_out.csv',index=False,float_format='%.6g')
# ---- 3. cluster LOETF (A 정의: Spearman, average, 1-ρ, maxclust k)
def clus(cols,k):
    S=Rt[cols].corr('spearman'); D=(1-S).to_numpy(copy=True); np.fill_diagonal(D,0); return pd.Series(fcluster(linkage(squareform(D,checks=False),'average'),k,'maxclust'),index=cols)
crow=[]
for base,cols in [('all20',TK),('dedup17',DED)]:
    for k in [3,5,8]:
        b=clus(cols,k)
        for t in cols:
            rest=[c for c in cols if c!=t]; c=clus(rest,k); crow.append(dict(base=base,k=k,dropped=t,scope=SC[t],ari_vs_full_restricted=ari(b[rest],c[rest])))
CL=pd.DataFrame(crow); CL.to_csv(f'{OUT}/cluster_leave_one_etf_out.csv',index=False,float_format='%.6g')
print(pd.DataFrame(summ)[['group','window','estimator','mean','p99_abs_daily_change','max_abs_daily_change','spearman_vs_vol20']+[f'jump_{a}' for a in ANC]].round(3).to_string())
P=pd.DataFrame(rows); print(P.groupby('base').apply(lambda g:g[~g.dropped.str.startswith('scaler')&(g.dropped!='(none)')].d_pc1.agg(['min','max'])).round(4)); print(P[P.dropped.str.startswith('scaler')|(P.dropped=='(none)')].round(3).to_string())
print(CL.groupby(['base','k']).ari_vs_full_restricted.agg(['min','median','mean']).round(3)); print(CL[CL.ari_vs_full_restricted<0.8].round(3).to_string())
