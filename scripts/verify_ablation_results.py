#!/usr/bin/env python3
"""Read-only, relocatable verification from distributed predictions; no training.

Requires NumPy, pandas and SciPy. Saved execution freezes are historical identities; distributed data files
are verified with DATA_MANIFEST.json.
"""
from pathlib import Path
from collections import defaultdict
import hashlib, json, math
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'artifacts/revision_2026_09_03'
ROLES={'icbhi2017':['validation_select','calibration','locked_test'], 'sprsound2022':['validation_select','calibration','locked_inter_test','locked_intra_test']}
PRIMARY={'icbhi2017':'locked_test','sprsound2022':'locked_inter_test'}
DIRS={'A':'analysis_a_corrected_2026_09_05','B':'analysis_b_path_corrected_2026_09_05','C':'analysis_c_2026_09_06'}
PREDS={'legacy-mask':'legacy_mask','no-js':'no_js_path_corrected_2026_09_05','matched-lora':'matched_lora_2026_09_06'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):return json.loads(p.read_text())
def close(a,b,tol=1e-11):np.testing.assert_allclose(a,b,rtol=0,atol=tol)
def metric(y,p):
    z=p>=.5;pos=y==1;neg=~pos;tp=sum(z&pos);tn=sum(~z&neg);fp=sum(z&neg);fn=sum(~z&pos)
    order=np.argsort(p,kind='stable');ranks=np.empty(len(p),float);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and p[order[j]]==p[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    auc=(ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())
    return {'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'average_score':.5*(tp/(tp+fn)+tn/(tn+fp)),
            'accuracy':(tp+tn)/len(y),'macro_f1':.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn)), 'auroc':auc}
def probmetric(y,p):
    p=np.clip(p,1e-6,1-1e-6);ix=np.argsort(p,kind='stable')
    return [-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)),np.mean((p-y)**2),
            sum(len(x)*abs(p[x].mean()-y[x].mean())/len(y) for x in np.array_split(ix,15))]
def coefficients(y,p):
    p=np.clip(p,1e-6,1-1e-6);x=np.column_stack([np.ones(len(p)),np.log(p)-np.log1p(-p)])
    fun=lambda b:np.mean(np.logaddexp(0,x@b)-y*(x@b))
    jac=lambda b:x.T@(expit(x@b)-y)/len(y)
    r=minimize(fun,[0.,0.],jac=jac,method='BFGS',options={'gtol':1e-10,'maxiter':1000})
    assert max(abs(jac(r.x)))<1e-7
    return r.x
def path(fam,seed,dom,role):
    if fam!='exact':return B/'predictions'/PREDS[fam]/f'seed{seed}_{dom}_{role}.csv'
    if role=='validation_select':return ROOT/f'runs/gate11a_exactmask_fullft_seed{seed}'/f'best_validation_predictions_{dom}.csv'
    sub='calibration_predictions' if role=='calibration' else 'locked_predictions'
    return ROOT/'artifacts/post_gate11a'/sub/f'fullft_seed{seed}_{dom}_{role}.csv'
def load(fam,seed,dom,role,manifest):
    df=pd.read_csv(path(fam,seed,dom,role),dtype={'patient_id':str}).sort_values('sample_id').reset_index(drop=True)
    ref=manifest[(manifest.dataset==dom)&(manifest.protocol_role==role)].sort_values('sample_id').reset_index(drop=True)
    assert len(df)==len(ref) and not df.sample_id.duplicated().any()
    for k in ['sample_id','patient_id','dataset','protocol_role']:assert df[k].astype(str).equals(ref[k].astype(str))
    assert np.array_equal(df.target,ref.binary_label_id)
    p=df.probability_1.to_numpy();assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    assert np.array_equal(df.prediction,p>=.5)
    return df
def boot(left,right,generator):
    patients=left.patient_id.to_numpy();unique=np.unique(patients); y=left.target.to_numpy();a=left.probability_1.to_numpy()>=.5;b=right.probability_1.to_numpy()>=.5
    counts=[]
    for patient in unique:
        ix=patients==patient;pos=ix&(y==1);neg=ix&(y==0)
        counts.append([sum(pos&a),sum(pos&b),sum(neg&~a),sum(neg&~b),sum(pos),sum(neg)])
    totals=np.asarray(counts)[generator.choice(len(unique),(4000,len(unique)),replace=True)].sum(axis=1)
    assert np.all(totals[:,4:]>0)
    # Match the original difference-of-AS arithmetic, rather than subtracting counts first.
    return .5*(totals[:,0]/totals[:,4]+totals[:,2]/totals[:,5])-.5*(totals[:,1]/totals[:,4]+totals[:,3]/totals[:,5])
def main():
    inventory=readj(ROOT/'DATA_MANIFEST.json')
    for name,row in inventory['files'].items():assert sha(ROOT/name)==row['sha256'],name
    assert not any(p.suffix.lower() in {'.wav','.flac','.mp3','.pt','.pth','.safetensors'} for p in ROOT.rglob('*') if p.is_file())
    mf=ROOT/'data/manifests/cross_domain_binary.csv';assert sha(mf)=='2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419'
    manifest=pd.read_csv(mf,dtype={'patient_id':str});assert len(manifest)==15987 and manifest.patient_id.nunique()==414
    for dom,roles in ROLES.items():
        use=['train_fit']+[r for r in roles if r!='locked_intra_test']
        sets=[set(manifest[(manifest.dataset==dom)&(manifest.protocol_role==r)].patient_id) for r in use]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:assert not a&b
    tables=0;cells=0;prob_cells=0;fits=0;boot_draws=0;ensembles={}
    for scope,folder in DIRS.items():
        result=readj(B/folder/f'analysis_{scope.lower()}_final.json');table=pd.read_csv(B/folder/'classification_metrics.csv')
        seedlist=[20260729,20260731] if scope=='A' else [20260729,20260730,20260731]
        other={'A':'legacy-mask','B':'no-js','C':'matched-lora'}[scope]
        for fam in ['exact',other]:
            for dom,roles in ROLES.items():
                for role in roles:
                    frames=[load(fam,s,dom,role,manifest) for s in seedlist]
                    for seed,df in zip(seedlist,frames):
                        row=table[(table.family==fam)&(table.member==f'seed{seed}')&(table.dataset==dom)&(table.role==role)].iloc[0]
                        for key,v in metric(df.target.to_numpy(),df.probability_1.to_numpy()).items():close(row[key],v);cells+=1
                        tables+=1
                    en=frames[0].copy();en.probability_1=np.mean(np.column_stack([x.probability_1 for x in frames]),axis=1)
                    ensembles[scope,fam,dom,role]=en
                    ensemble_name='matched_completer_ensemble' if scope=='A' else 'ensemble'
                    row=table[(table.family==fam)&(table.member==ensemble_name)&(table.dataset==dom)&(table.role==role)].iloc[0]
                    assert row['samples']==len(en) and row['patients']==en.patient_id.nunique()
                    for key,v in metric(en.target.to_numpy(),en.probability_1.to_numpy()).items():close(row[key],v);cells+=1
                    tables+=1
        diffs=[]
        for (dom,role),sequence in zip(PRIMARY.items(),np.random.SeedSequence(20260729).spawn(2)):
            left=ensembles[scope,'exact',dom,role];right=ensembles[scope,other,dom,role]
            vals=boot(left,right,np.random.default_rng(sequence));diffs.append(vals);boot_draws+=len(vals)
            key={'A':'exact_minus_legacy_average_score','B':'js_on_minus_no_js','C':'fullft_minus_matched_lora'}[scope]
            rec=result['exact_minus_legacy' if scope=='A' else 'primary_contrast'][dom][key]
            point=metric(left.target.to_numpy(),left.probability_1.to_numpy())['average_score']-metric(right.target.to_numpy(),right.probability_1.to_numpy())['average_score']
            close(rec['estimate'],point);close([rec['lower'],rec['upper']],np.quantile(vals,[.025,.975]))
            if scope!='A':close(vals,np.load(B/folder/'classification_bootstrap_draws.npz')[dom],1e-14)
        equal=np.mean(np.stack(diffs),axis=0);rec=result['exact_minus_legacy' if scope=='A' else 'primary_contrast']['equal_database_mean'];close([rec['lower'],rec['upper']],np.quantile(equal,[.025,.975]));boot_draws+=4000
        for _,r in pd.read_csv(B/folder/'calibration_metrics.csv').iterrows():
            df=ensembles[scope,r['family'],r['dataset'],r['role']];y=df.target.to_numpy();p=df.probability_1.to_numpy()
            if r['scale']=='temperature':
                p=np.clip(p,1e-6,1-1e-6);p=expit((np.log(p)-np.log1p(-p))/r['temperature'])
            close([r[k] for k in ['negative_log_likelihood','brier_score','equal_frequency_ece']],probmetric(y,p),1e-9);prob_cells+=3
            close([r.calibration_intercept,r.calibration_slope],coefficients(y,p),2e-5);fits+=1
    assert tables==154 and cells==924 and prob_cells==252 and fits==84 and boot_draws==36000
    a=readj(B/DIRS['A']/'analysis_a_final.json');assert a['failed_legacy_seed_retained']==20260730 and not a['failed_seed_used_for_classification_or_calibration']
    s=readj(B/'stability_panel_2026_09_06/stability_panel.json')
    assert len(s['placement_records'])==535 and len(s['backward_records'])==60
    for fam,expected,back in [('exact',(321,321),(30,30)),('legacy-mask',(196,214),(21,30))]:
        pred=[r for r in s['placement_records'] if r['family']==fam];rr=[r for r in s['backward_records'] if r['family']==fam]
        assert (sum(r['probabilities_finite'] for r in pred),len(pred))==expected
        assert (sum(r['all_finite'] for r in rr),len(rr))==back
    res=readj(B/'resource_profile_2026_09_06/resource_profile.json')
    for r in res['inference']:
        vals=np.asarray(r['timings_seconds']);assert len(vals)==20 and np.isfinite(vals).all()
        close(r['batch_latency_median_seconds'],np.median(vals));close(r['batch_latency_p95_seconds'],np.quantile(vals,.95))
        close(r['per_event_latency_median_seconds'],np.median(vals)/r['batch_size']);close(r['throughput_events_per_second'],r['batch_size']/np.median(vals))
    print(json.dumps({'status':'DERIVED_DATA_VERIFIED','distributed_files':len(inventory['files']),'manifest_events':15987,
      'classification_rows':tables,'classification_cells':cells,'probability_cells':prob_cells,'independent_joint_point_fits':fits,
      'paired_classification_draws':boot_draws,'stability_predictions':535,'backward_pairs':60,'resource_settings':20,
      'scope':'read-only saved-data reconstruction; no training/inference or new analysis selection'},indent=2))
if __name__=='__main__':main()
