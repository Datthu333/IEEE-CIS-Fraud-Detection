"""Week 1 IEEE-CIS EDA using training data only.

Input: ../cleaned_data/train_cleaned.csv (merged training transaction + identity data).
This script never opens any test file and never writes to the input dataset.
Run with the bundled Python runtime; dependencies are listed in requirements.txt.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
VENDOR=ROOT/'vendor'
if VENDOR.exists(): sys.path.insert(0,str(VENDOR))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA=PROJECT/'cleaned_data'/'train_cleaned.csv'
CHARTS=ROOT/'charts'
CHUNK_SIZE=20000
sns.set_theme(style='whitegrid',context='talk',palette='deep')
plt.rcParams.update({'figure.dpi':140,'savefig.dpi':300,'font.family':'DejaVu Sans','axes.titleweight':'bold','axes.labelcolor':'#243447','text.color':'#243447','axes.edgecolor':'#ccd5df','grid.color':'#e7edf3'})

def wilson(k,n,z=1.96):
    if n==0:return (0.0,0.0)
    p=k/n; d=1+z*z/n; center=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.0,center-half),min(1.0,center+half)

def main():
    if not DATA.is_file(): raise FileNotFoundError(f'Training input not found: {DATA}')
    header=pd.read_csv(DATA,nrows=0).columns.tolist()
    if 'isFraud' not in header: raise ValueError('Training target is missing')
    if any('test' in str(c).lower() for c in header): raise ValueError('Unexpected test-like column name')
    identity_cols=[c for c in header if c.startswith('id_') or c in ('DeviceType','DeviceInfo')]
    transaction_features=[c for c in header if c not in identity_cols and c not in ('TransactionID','isFraud')]
    missing=pd.Series(0,index=header,dtype='int64')
    rows=0; target_counts={0:0,1:0}; identity_coverage={0:{'present':0,'absent':0},1:{'present':0,'absent':0}}
    selected=[]; product_counts={}; card4_counts={}; card6_counts={}; email_counts={}
    for chunk in pd.read_csv(DATA,chunksize=CHUNK_SIZE,low_memory=False):
        rows+=len(chunk); missing=missing.add(chunk.isna().sum().reindex(header,fill_value=0),fill_value=0).astype('int64')
        labels=chunk['isFraud'].astype('int64')
        vc=labels.value_counts()
        for cls in (0,1): target_counts[cls]+=int(vc.get(cls,0))
        present=chunk[identity_cols].notna().any(axis=1)
        for cls in (0,1):
            m=labels==cls; identity_coverage[cls]['present']+=int((m&present).sum()); identity_coverage[cls]['absent']+=int((m&~present).sum())
        for col,agg in [('ProductCD',product_counts),('card4',card4_counts),('card6',card6_counts),('P_emaildomain',email_counts)]:
            c=chunk.groupby([col,'isFraud'],dropna=False).size()
            for (cat,y),n in c.items():
                key=(str(cat) if pd.notna(cat) else 'Missing',int(y)); agg[key]=agg.get(key,0)+int(n)
        selected.append(chunk[['TransactionID','isFraud','TransactionAmt','ProductCD','card4','card6','P_emaildomain']].copy())
    df=pd.concat(selected,ignore_index=True)
    df['TransactionAmt']=pd.to_numeric(df['TransactionAmt'],errors='coerce')
    if rows!=590540: raise ValueError(f'Expected 590540 training rows, found {rows}')
    if set(df['isFraud'].dropna().unique())!={0,1}: raise ValueError('Unexpected target labels')
    total=target_counts[0]+target_counts[1]; fraud_rate=target_counts[1]/total
    amount=df['TransactionAmt'].dropna(); amount_by_class={}
    for cls,g in df.groupby('isFraud'):
        a=g['TransactionAmt'].dropna()
        amount_by_class[str(int(cls))]={k:float(a.quantile(q)) for k,q in [('p25',.25),('median',.5),('p75',.75),('p90',.9),('p99',.99)]}
        amount_by_class[str(int(cls))].update({'mean':float(a.mean()),'n':int(a.size)})
    amount_nonpositive=int((amount<=0).sum()); amount_missing=int(df['TransactionAmt'].isna().sum())
    missing_pct=(missing/rows*100).round(6)
    miss_table=pd.DataFrame({'column':header,'missing_count':missing.values,'missing_pct':missing_pct.values})
    miss_table=miss_table.sort_values(['missing_pct','column'],ascending=[False,True])
    miss_table.to_csv(ROOT/'missingness_by_column.csv',index=False)

    def table(agg):
        rows=[]
        for (cat,y),n in agg.items(): rows.append({'category':cat,'isFraud':y,'count':n})
        frame=pd.DataFrame(rows)
        if frame.empty:return frame
        wide=frame.pivot_table(index='category',columns='isFraud',values='count',aggfunc='sum',fill_value=0).rename(columns={0:'legitimate',1:'fraudulent'})
        for c in ('legitimate','fraudulent'):
            if c not in wide:wide[c]=0
        wide['total']=wide.legitimate+wide.fraudulent
        wide['fraud_rate']=wide.fraudulent/wide.total
        ci=wide.apply(lambda r:wilson(int(r.fraudulent),int(r.total)),axis=1)
        wide['ci_low']=[x[0] for x in ci];wide['ci_high']=[x[1] for x in ci]
        return wide.reset_index().sort_values('total',ascending=False)
    product=table(product_counts); card4=table(card4_counts); card6=table(card6_counts); email=table(email_counts)
    product.to_csv(ROOT/'productcd_fraud_rates.csv',index=False); card4.to_csv(ROOT/'card4_fraud_rates.csv',index=False);card6.to_csv(ROOT/'card6_fraud_rates.csv',index=False);email.to_csv(ROOT/'email_domain_fraud_rates.csv',index=False)
    n_id=len(identity_cols); trans_missing=float(missing_pct[transaction_features].mean()); ident_missing=float(missing_pct[identity_cols].mean())
    top_missing=miss_table.head(20).to_dict(orient='records')
    product_top=product.sort_values(['fraud_rate','total'],ascending=[False,False]).iloc[0]
    product_vol=product.sort_values('total',ascending=False).iloc[0]
    card4_eligible=card4[card4.total>=1000].sort_values('fraud_rate',ascending=False)
    card6_eligible=card6[card6.total>=1000].sort_values('fraud_rate',ascending=False)
    findings=[
      f"The training set contains {total:,} transactions: {target_counts[1]:,} fraudulent ({fraud_rate:.2%}) and {target_counts[0]:,} legitimate ({target_counts[0]/total:.2%}). This is a strongly imbalanced target distribution.",
      f"TransactionAmt is right-skewed: the overall median is {amount.median():.2f}, the mean is {amount.mean():.2f}, and the 99th percentile is {amount.quantile(.99):.2f}. The plot uses a log10 amount axis so the long tail and dense low-value region are both visible.",
      f"Fraudulent transaction amount median is {amount_by_class['1']['median']:.2f} versus {amount_by_class['0']['median']:.2f} for legitimate transactions. This is a descriptive difference and does not establish a cause.",
      f"ProductCD category {product_top['category']} has the highest observed fraud rate ({product_top['fraud_rate']:.2%}, n={int(product_top['total']):,}); the largest category by volume is {product_vol['category']} (n={int(product_vol['total']):,}). Rates are accompanied by Wilson 95% intervals and should be read with category sizes.",
      f"Average missingness is {ident_missing:.1f}% across the {n_id} identity columns and {trans_missing:.1f}% across the {len(transaction_features)} transaction feature columns in the merged training table. Missingness includes transactions without a linked identity record.",
      f"Identity attributes are present for {identity_coverage[1]['present']:,}/{target_counts[1]:,} fraudulent transactions ({identity_coverage[1]['present']/target_counts[1]:.2%}) and {identity_coverage[0]['present']:,}/{target_counts[0]:,} legitimate transactions ({identity_coverage[0]['present']/target_counts[0]:.2%}), where presence means at least one joined identity field is nonmissing. This difference is descriptive only.",
    ]
    if len(card4_eligible): findings.append(f"Among card4 categories with at least 1,000 transactions, {card4_eligible.iloc[0]['category']} has the highest observed fraud rate ({card4_eligible.iloc[0]['fraud_rate']:.2%}, n={int(card4_eligible.iloc[0]['total']):,}).")
    if len(card6_eligible): findings.append(f"Among card6 categories with at least 1,000 transactions, {card6_eligible.iloc[0]['category']} has the highest observed fraud rate ({card6_eligible.iloc[0]['fraud_rate']:.2%}, n={int(card6_eligible.iloc[0]['total']):,}).")

    # Chart 1: class balance.
    fig,ax=plt.subplots(figsize=(8.8,5.5)); labels=['Legitimate','Fraudulent']; vals=[target_counts[0],target_counts[1]]; colors=['#3977a8','#e45756']
    bars=ax.bar(labels,vals,color=colors,width=.58)
    ax.set_title('Fraud is a small share of training transactions',loc='left',pad=16);ax.set_ylabel('Transactions');ax.set_ylim(0,max(vals)*1.15)
    for b,n in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*.02,f'{n:,}\n{n/total:.2%}',ha='center',va='bottom',fontsize=12,fontweight='bold')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(CHARTS/'01_class_distribution.png',dpi=300,bbox_inches='tight');plt.close(fig)

    # Chart 2: per-class density on a log10 amount axis.
    positive=df[df.TransactionAmt>0].copy();positive['log10_amount']=np.log10(positive.TransactionAmt)
    fig,ax=plt.subplots(figsize=(10,6));
    for cls,color,label in [(0,'#3977a8','Legitimate'),(1,'#e45756','Fraudulent')]:
        vals=positive.loc[positive.isFraud==cls,'log10_amount'];sns.histplot(vals,bins=70,stat='density',element='step',fill=False,linewidth=2,color=color,label=label,ax=ax)
        med=np.log10(df.loc[df.isFraud==cls,'TransactionAmt'].median());ax.axvline(med,color=color,linestyle='--',alpha=.75)
    ax.set_title('Transaction amounts have a long right tail',loc='left',pad=16);ax.set_xlabel('Transaction amount (log10 scale; dashed lines show class medians)');ax.set_ylabel('Density');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(CHARTS/'02_amount_distribution.png',dpi=300,bbox_inches='tight');plt.close(fig)

    # Chart 3: most-missing fields, grouped by transaction or identity source.
    plotmiss=miss_table.head(20).copy();plotmiss['source']=plotmiss.column.apply(lambda c:'Identity' if c in identity_cols else ('Key/target' if c in ('TransactionID','isFraud') else 'Transaction'))
    fig,ax=plt.subplots(figsize=(10,8));sns.barplot(data=plotmiss,y='column',x='missing_pct',hue='source',dodge=False,palette={'Identity':'#e08e45','Transaction':'#3977a8','Key/target':'#718096'},ax=ax)
    ax.set_title('Identity fields dominate the highest missingness rates',loc='left',pad=16);ax.set_xlabel('Missing values (%)');ax.set_ylabel('');ax.legend(title='Field group',frameon=False,loc='center left',bbox_to_anchor=(1.01,.5));ax.set_xlim(0,103);ax.spines[['top','right']].set_visible(False);fig.tight_layout(rect=[0,0,.82,1]);fig.savefig(CHARTS/'03_missingness_top20.png',dpi=300,bbox_inches='tight');plt.close(fig)

    # Chart 4: ProductCD rate with Wilson 95% intervals.
    p=product.sort_values('fraud_rate');fig,ax=plt.subplots(figsize=(9,5.8));x=np.arange(len(p));rates=p.fraud_rate.to_numpy();lo=rates-p.ci_low.to_numpy();hi=p.ci_high.to_numpy()-rates
    ax.bar(x,rates,color='#4d90b8',width=.62);ax.errorbar(x,rates,yerr=[lo,hi],fmt='none',ecolor='#243447',capsize=4,lw=1.3)
    ax.set_xticks(x,p.category);ax.set_ylim(0,max(p.ci_high)*1.28);ax.set_title('Observed fraud rate varies across ProductCD',loc='left',pad=16);ax.set_ylabel('Fraudulent transactions (%)');ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.0%}'));ax.set_xlabel('ProductCD category')
    for i,r in p.reset_index(drop=True).iterrows():ax.text(i,r.fraud_rate+max(p.ci_high)*.08,f"n={int(r.total):,}",ha='center',fontsize=9)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(CHARTS/'04_productcd_fraud_rate.png',dpi=300,bbox_inches='tight');plt.close(fig)

    # Chart 5: card brand/type fraud rates; omit rare values from chart, not from tables.
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,frame,title in [(axes[0],card4,'Card brand (card4)'),(axes[1],card6,'Card type (card6)')]:
        q=frame[frame.total>=1000].sort_values('fraud_rate')
        y=np.arange(len(q));ax.barh(y,q.fraud_rate,color='#6a9f58');ax.errorbar(q.fraud_rate,y,xerr=[q.fraud_rate-q.ci_low,q.ci_high-q.fraud_rate],fmt='none',ecolor='#243447',capsize=3,lw=1)
        ax.set_yticks(y,q.category);ax.set_xlim(0,max(q.ci_high.max()*1.22,.01));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.0%}'));ax.set_title(title,loc='left');ax.set_xlabel('Observed fraud rate')
        for yi,(_,r) in enumerate(q.iterrows()):ax.text(r.ci_high+max(q.ci_high.max()*.03,.002),yi,f"n={int(r.total):,}",va='center',fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Card categories show descriptive fraud-rate differences',x=.07,ha='left',fontweight='bold',fontsize=18);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(CHARTS/'05_card_category_rates.png',dpi=300,bbox_inches='tight');plt.close(fig)

    summary={'input_file':str(DATA.resolve()),'input_scope':'training data only; no test files opened','rows':rows,'columns':len(header),'target_counts':{str(k):v for k,v in target_counts.items()},'fraud_rate':fraud_rate,'amount_summary':{'overall':{'n':int(amount.size),'min':float(amount.min()),'median':float(amount.median()),'mean':float(amount.mean()),'p75':float(amount.quantile(.75)),'p90':float(amount.quantile(.90)),'p99':float(amount.quantile(.99)),'max':float(amount.max()),'missing_count':amount_missing,'nonpositive_count':amount_nonpositive},'by_class':amount_by_class},'missingness':{'average_transaction_feature_pct':trans_missing,'average_identity_feature_pct':ident_missing,'top_20':top_missing},'identity_presence_by_class':{str(k):v for k,v in identity_coverage.items()},'productcd':product.to_dict(orient='records'),'card4':card4.to_dict(orient='records'),'card6':card6.to_dict(orient='records'),'email_domain_top_volume':email.head(12).to_dict(orient='records'),'findings':findings,'charts':[str((CHARTS/f).resolve()) for f in sorted(p.name for p in CHARTS.glob('*.png'))]}
    (ROOT/'eda_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Week 1 EDA findings','',f"Input: `{DATA.resolve()}`. This analysis uses only the cleaned training table ({rows:,} rows, {len(header)} columns). No test file was opened.",'','## Findings','']+[f'{i}. {x}' for i,x in enumerate(findings,1)]+['','## Visuals','']
    captions=[('01_class_distribution.png','The bars compare legitimate and fraudulent transaction counts and percentages. The imbalance matters because overall accuracy alone could hide poor fraud detection.'),('02_amount_distribution.png','The two curves compare transaction amounts by target on a log10 scale. Dashed lines mark each class median; the scale makes the long tail easier to see.'),('03_missingness_top20.png','The bars show the 20 columns with the largest missing percentages after the identity merge. This matters because absent identity information is common and should not be mistaken for a failed transaction row.'),('04_productcd_fraud_rate.png','The bars show the share of fraudulent rows within each ProductCD category. Error bars show Wilson 95% intervals, and labels give category sizes; rates are descriptive, not causal.'),('05_card_category_rates.png','The panels compare observed fraud rates by card brand and card type for categories with at least 1,000 rows. Rare categories remain in the CSV summary but are omitted from these charts for readability.')]
    for fn,caption in captions:md+= [f'### {fn.replace("_"," ").replace(".png","")}',f'![{caption}](charts/{fn})','',caption,'']
    (ROOT/'findings.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'rows':rows,'columns':len(header),'fraud_rate':fraud_rate,'charts':summary['charts'],'findings':len(findings)},indent=2))

if __name__=='__main__':main()
