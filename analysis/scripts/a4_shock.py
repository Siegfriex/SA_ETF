import pandas as pd, numpy as np
A='/home/sieg/projects-wsl/hongik_univ_26_2/SA/.worktrees/SA_ETF/a/'
O=A+'analysis/shock/'
r=pd.read_csv('/home/sieg/projects-wsl/hongik_univ_26_2/SA/ETF_EDA_SCAFFOLD/data/interim/panel_logret.csv',index_col=0,parse_dates=True)
r.loc[['2019-03-14','2019-03-15'],'261240']=np.nan
T=list(r.columns)
tax=pd.read_csv(A+'analysis/taxonomy_draft.csv',dtype={'ticker':str}).set_index('ticker')['asset_scope'].to_dict()
med=r.rolling(60,min_periods=40).median().shift(1)
mad=r.rolling(60,min_periods=40).apply(lambda x: np.nanmedian(np.abs(x-np.nanmedian(x))),raw=True).shift(1)
rz=(r-med)/(1.4826*mad)
rz.to_csv(O+'rz60.csv')
EQ=['069500','102110','229200','091230','091170','091180','305540','143860','117680','161510','219480','133690','192090']
K2=['069500','102110','122630','114800']
sgn=np.sign(r).copy(); sgn['114800']=-sgn['114800']
ref=np.sign(r['069500'])
ext=rz.abs()>3
same=ext & sgn.eq(ref,axis=0) & (ref!=0).values[:,None]
opp=ext & sgn.eq(-ref,axis=0) & (ref!=0).values[:,None]
d=pd.DataFrame({'xs_mean':r.mean(1),'xs_disp':r.std(1),'xs_disp_eq':r[EQ].std(1),
 'n_absrz3':ext.sum(1),'n_absrz3_dedup':ext.drop(columns=K2[1:]).sum(1),'n_same_dir':same.sum(1),'n_opposite_dir':opp.sum(1)})
d=d[rz.notna().sum(1)>0]
d.index.name='date'; d.to_csv(O+'daily.csv')
print('N days',len(d))
print(d.describe(percentiles=[.5,.9,.95,.99]).round(5).T.to_string())
print(d.groupby(d.index.year)[['xs_disp','xs_disp_eq','n_absrz3']].median().round(5))
print('n_absrz3 dist', d.n_absrz3.value_counts().sort_index().to_dict())
for k in [3,5,8]: print('k',k,(d.n_absrz3>=k).sum(),'dedup',(d.n_absrz3_dedup>=k).sum())
# quadrants
am=d.xs_mean.abs(); hm=am>am.median(); hd=d.xs_disp>d.xs_disp.median()
q=pd.crosstab(hm,hd); print('quadrant (rows |mean| high, cols disp high)\n',q)
am90=am>am.quantile(.9); hd90=d.xs_disp>d.xs_disp.quantile(.9); lo50=d.xs_disp<d.xs_disp.median()
print('sync: |mean|>q90 & disp<median', (am90&lo50).sum(), ' diverge: |mean|<median & disp>q90', ((~hm)&hd90).sum(), ' both>q90',(am90&hd90).sum())
print('top disp days\n', d.sort_values('xs_disp',ascending=False).head(10).round(4).to_string())
# top15
scopes=['domestic_equity','foreign_equity','bond','commodity','fx']
rows=[]
for dt in d.sort_values(['n_absrz3','n_absrz3_dedup'],ascending=False).head(15).index:
    row={'date':dt.date(),'n':d.n_absrz3[dt],'n_dedup':d.n_absrz3_dedup[dt],'r069500_pct':round(100*r.loc[dt,'069500'],2),'n_same':d.n_same_dir[dt],'n_opp':d.n_opposite_dir[dt]}
    for s in scopes:
        ts=[t for t in T if tax[t]==s]
        e=[t for t in ts if ext.loc[dt,t]]
        row[s]=f"{len(e)}/{len(ts)} S{sum(same.loc[dt,t] for t in e)} O{sum(opp.loc[dt,t] for t in e)}"
    row['opp_list']=','.join([t for t in T if opp.loc[dt,t]])
    rows.append(row)
top=pd.DataFrame(rows); top.to_csv(O+'top15_common.csv',index=False); print(top.to_string())
# participation by scope on k>=5 days
for k in [5]:
    days=d.index[d.n_absrz3>=k]
    pr=[]
    for s in scopes:
        ts=[t for t in T if tax[t]==s]
        part=ext.loc[days,ts].values.mean()
        sm=same.loc[days,ts].values.sum(); op=opp.loc[days,ts].values.sum()
        pr.append({'scope':s,'n_etf':len(ts),'k':k,'n_days':len(days),'part_rate':round(part,3),'n_same':int(sm),'n_opp':int(op),'dir_score':round((sm-op)/max(sm+op,1),3)})
    pr=pd.DataFrame(pr); pr.to_csv(O+'scope_participation_k5.csv',index=False); print(pr.to_string())
# idio
q95=r.abs().quantile(.95)
id_rows=[]
for dt in rz.index:
    for t in T:
        z=rz.loc[dt,t]
        if pd.notna(z) and abs(z)>4:
            oth=rz.loc[dt].drop(t).abs().median()
            if oth<1:
                id_rows.append({'date':dt.date(),'ticker':t,'r_pct':round(100*r.loc[dt,t],3),'rz':round(z,2),'others_med_absrz':round(oth,2),
                 'econ_q95':bool(abs(r.loc[dt,t])>=q95[t]),'econ_q95_or_50bp':bool(abs(r.loc[dt,t])>=q95[t] or abs(r.loc[dt,t])>=0.005),'scale_artifact':t=='153130'})
idf=pd.DataFrame(id_rows); idf.to_csv(O+'idio.csv',index=False)
g=idf.groupby('ticker').agg(n=('rz','size'),n_econ_q95=('econ_q95','sum'),n_econ_q95_or_50bp=('econ_q95_or_50bp','sum'))
g['q95_abs_r_pct']=(100*q95).round(3)
print(g.sort_values('n',ascending=False).to_string()); g.to_csv(O+'idio_by_etf.csv')
print('total',len(idf),'excl153130',(idf.ticker!='153130').sum(),'econ_q95 excl',((idf.ticker!='153130')&idf.econ_q95).sum())
for t,gg in idf[idf.ticker!='153130'].groupby('ticker'):
    print(t, gg.reindex(gg.rz.abs().sort_values(ascending=False).index).head(3)[['date','r_pct','rz','others_med_absrz','econ_q95']].values.tolist())
print('153130 top', idf[idf.ticker=='153130'].sort_values('rz').head(3).values.tolist())
for dt,t in [('2024-10-02','192090'),('2020-04-22','261220'),('2026-01-30','132030'),('2021-10-05','143860'),('2021-10-29','148070'),('2024-01-30','153130')]:
    print('case',dt,t,'r%',round(100*r.loc[dt,t],2),'rz',round(rz.loc[dt,t],2),'othmed',round(rz.loc[dt].drop(t).abs().median(),2),'q95%',round(100*q95[t],3))
# anchors
anc=[('A1','2020-03-19','2020-03-24'),('A2','2024-08-05','2024-08-05'),('A3','2026-03-04','2026-03-04'),('A4','2026-07-28','2026-08-03')]
absx=r.abs()>0.03
for a,s,e in anc:
    w=d.loc[s:e]; print('\n',a,s,e); 
    for dt in w.index:
        print(dt.date(),'n',w.n_absrz3[dt],'dd',w.n_absrz3_dedup[dt],'same',w.n_same_dir[dt],'opp',w.n_opposite_dir[dt],'abs3',int(absx.loc[dt].sum()),'r069500%',round(100*r.loc[dt,'069500'],2),
          'ext:',[t for t in T if ext.loc[dt,t]],'opp:',[t for t in T if opp.loc[dt,t]])
        print('   r% nonKR', {t:round(100*r.loc[dt,t],2) for t in ['219480','133690','192090','148070','153130','132030','261220','261240']})
print('MAD60 069500 bp:', (1.4826*mad['069500']*1e4).loc['2026-05-25':'2026-08-05'].round(0).iloc[::3].to_dict())
print('rz 069500 07-28..08-03', rz['069500'].loc['2026-07-28':'2026-08-03'].round(2).to_dict())
# rolling corr vs disp
EQ1=EQ[:10]+EQ[10:]  # equity 1x block
C=r[EQ].rolling(120,min_periods=100).corr()
mc=C.groupby(level=0).apply(lambda m: (m.values[np.triu_indices(len(EQ),1)]).mean())
dd=d.xs_disp_eq.rolling(120,min_periods=100).mean()
x=pd.concat([mc.rename('mcorr'),dd.rename('disp')],axis=1).dropna()
x.to_csv(O+'rolling120_corr_disp.csv')
print('corr levels', x.corr().iloc[0,1].round(3), 'spearman', x.corr('spearman').iloc[0,1].round(3))
x2=x[x.index<'2026-01-01']; print('pre2026', x2.corr().iloc[0,1].round(3), x2.corr('spearman').iloc[0,1].round(3))
print(x.resample('YE').last().round(4))
print('vol 069500 by yr %', (r['069500'].groupby(r.index.year).std()*100).round(3).to_dict())
print('xs_disp_eq ann median', (d.xs_disp_eq.groupby(d.index.year).median()*100).round(3).to_dict())
