"""Four-source, three-seed expansion of the unchanged composite pilot.

Classification uses historical development holdouts, not original sealed tests.
Nine table methods plus explicit system controls. No scientific tuning.
"""
import argparse
import copy
import gc
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import e2_composite_corruption_pilot_20261118 as pilot

b, expert = pilot.b, pilot.expert
ROOT = pilot.ROOT
NAME = "e2_composite_four_class_20261118"
OUT = ROOT / "results" / NAME
CFG = ROOT / "configs" / (NAME + ".json")
PARENT = ROOT / "results/e2_composite_corruption_pilot_20261118"
FROZEN = pilot.FROZEN
read, write, sha, key = pilot.read, pilot.write, pilot.sha, pilot.key
METHODS = ("raw", "standard", "autofeat", "caafe", "llmfe", "classical_clean",
           "lm_cleaning_simplified", "clean_then_openfe")
LM_METHODS = ("caafe", "llmfe", "lm_cleaning_simplified")
CODES = ("german_credit", "default_credit", "bank_marketing", "qsar-biodeg")
SEEDS = (42, 123, 456)


def records():
    return read(OUT / "cases.json")


def normalized_records():
    return [r for r in records() if r["tag"].startswith("N_")]


def configure():
    pilot.OUT, pilot.CFG, pilot.NAME = OUT, CFG, NAME
    pilot.records, pilot.normalized_records = records, normalized_records
    pilot.configure()
    pilot.__file__ = __file__
    b.cases = records


def remap(value):
    if isinstance(value,str):
        return value.replace(str(PARENT),str(OUT))
    if isinstance(value,dict):
        return {k:remap(v) for k,v in value.items()}
    if isinstance(value,list):
        return [remap(v) for v in value]
    if isinstance(value,tuple):
        return tuple(remap(v) for v in value)
    return value


def prepare():
    import joblib
    import numpy as np
    import pandas as pd
    pilot.check_time()
    assert not (OUT / "prepared.json").exists()
    version = read(FROZEN / "CONFIRMED_VERSION.json")
    for p,digest in version["implementation"].items():
        assert sha(ROOT/p)==digest
    assert read(PARENT/"controller_complete.json")["exit"]==0
    shutil.copytree(FROZEN/"runtime",OUT/"runtime",ignore=shutil.ignore_patterns("__pycache__"))
    config=read(FROZEN/"run_config.json")
    config["output_dir"]=str(OUT)
    config["field_metadata"]={}
    write(OUT/"run_config.json",config)
    rt,_=pilot.bind()
    from taskdq.e2_parameterized import registry
    protected,entries,audits,inherited={},[],[],[]
    parent_config=read(PARENT/"run_config.json")
    for code in CODES:
        if code in CODES[:2]:
            old=pilot.OLD/code
            srcpath=pilot.OLD_RESULTS/"units"/code/"42/search/source.joblib"
            source=joblib.load(srcpath)
            manifest=read(old/"manifest.json")
            forbidden=read(old/"split.json")
            forbidden_ids=set(forbidden["confirmation_row_ids"]+forbidden["final_test_row_ids"])
            for path in [old/"confirmation.joblib",old/"final_test.joblib",old/"split.json",old/"development.joblib",old/"manifest.json",srcpath]:
                protected[str(path.relative_to(ROOT))]=sha(path)
            assert not set(source["x"].index)&forbidden_ids
            metadata=read(pilot.OLD_RESULTS/"run_config.json")["field_metadata"][code]
            numeric=["credit_amount","duration"] if code=="german_credit" else ["LIMIT_BAL",*["BILL_AMT"+str(i) for i in range(1,7)],*["PAY_AMT"+str(i) for i in range(1,7)]]
            if code=="german_credit":
                source["x"]=source["x"].rename(columns={"duration_months":"duration"})
                metadata=copy.deepcopy(metadata)
                metadata["columns"]["duration"]=metadata["columns"].pop("duration_months")+"; official month duration, shared alias duration."
            categories=manifest["categorical"] if code=="german_credit" else ["SEX","EDUCATION","MARRIAGE"]
            source_rows=manifest["source_rows"]
        else:
            old_name="e2_openml_main_20261105" if code=="bank_marketing" else "e2_unified_independent_20261107"
            old=ROOT/"data"/old_name/"prepared"/code
            srcpath=ROOT/"results"/old_name/"units"/code/"42/final/source.joblib"
            source=joblib.load(srcpath)
            development=joblib.load(old/"42/development.joblib")
            assert source["x"].equals(development["x"]) and source["y"].equals(development["y"])
            split=read(old/"42/split.json")
            assert not set(source["x"].index)&set(split["test_row_ids"])
            metadata=read(ROOT/"results"/old_name/"run_config.json").get("field_metadata",{}).get(code,
                dict(columns={c:"Original public source field: "+c for c in source["x"]},
                     numeric_type_hints=[],same_quantity_groups=[]))
            for path in [srcpath,old/"42/development.joblib",old/"42/sealed_test.joblib",old/"42/split.json",old/"manifest.json"]:
                protected[str(path.relative_to(ROOT))]=sha(path)
            source_rows=read(old/"manifest.json")["source_rows"]
            numeric=["balance"] if code=="bank_marketing" else ["SpMax_L","J_Dz(e)","SM6_L","SpMax_B(m)"]
            categories=[c for c in source["x"] if not pd.api.types.is_numeric_dtype(source["x"][c])]
            if code=="bank_marketing":
                # All selected groups remain in this old development pool.
                groups=source["groups"]
                ordered=sorted(groups.unique(),key=lambda v:key(["e2-four-bank-cap-v1",str(v)]))
                selected=[]
                for group in ordered:
                    ids=np.flatnonzero(groups.to_numpy()==group)
                    if len(selected)+len(ids)<=6000:
                        selected.extend(ids.tolist())
                    if len(selected)==6000:
                        break
                selected=np.sort(selected)
                assert len(selected)>5000
                for field in ("x","y","groups"):
                    source[field]=source[field].iloc[selected]
                assert "duration" not in source["x"]
        for column in categories:
            before=source["x"][column].astype("string")
            after=before.str.strip().str.casefold()
            assert before.nunique()==after.nunique()
            source["x"][column]=after.astype(object).where(after.notna(),np.nan)
        contracts=[dict(column=c,unit_column=c+"__export_unit") for c in numeric]
        partial=pilot.masked(source["x"],source["groups"],numeric)
        for seed in SEEDS:
            a,z=b.folds(source,5,seed)[0]
            a,z=np.sort(a),np.sort(z)
            assert not set(source["groups"].iloc[a])&set(source["groups"].iloc[z])
            train=pilot.export(partial.iloc[a],contracts,categories,groups=source["groups"])
            query=pilot.export(partial.iloc[z],contracts,categories,query=True)
            normalized=pilot.previous.mechanism.restore(train,contracts)
            normalized_query=pilot.previous.mechanism.restore(query,contracts)
            pd.testing.assert_frame_equal(normalized,partial.iloc[a],check_exact=False,rtol=1e-12,atol=1e-12)
            pd.testing.assert_frame_equal(normalized_query,partial.iloc[z],check_exact=False,rtol=1e-12,atol=1e-12)
            current=copy.deepcopy(source)
            for field in ("x","y","groups"):
                current[field]=source[field].iloc[a]
            _,support=b.folds(current,4,seed+11)[0]
            support=np.sort(support)
            rest=np.setdiff1d(np.arange(len(a)),support)
            sub=copy.copy(current)
            for field in ("x","y","groups"):
                sub[field]=current[field].iloc[rest]
            folds=[]
            for ia,iz in b.folds(sub,3,seed+17):
                ia,iz=np.sort(np.r_[support,rest[ia]]),np.sort(rest[iz])
                assert not set(current["groups"].iloc[ia])&set(current["groups"].iloc[iz])
                folds.append(dict(train=ia.tolist(),validation=iz.tolist()))
            public=(" Public transport definitions, identical for all methods: "+str(contracts)+
                ". Each observed value is canonical value multiplied by the number after times_ in its unit column. "
                "Training mixes factors1/1000 per row and column; validation/deployment factor100. "
                "Text category meaning is invariant to case and surrounding whitespace. Transport marker columns are not outcomes. "
                "5% of declared numeric cells may be missing; missing values cannot be recovered from metadata.")
            for prefix,x,q in (("R",train,query),("N",normalized,normalized_query)):
                tag=f"{prefix}_{code}__{seed}"
                unit=rt.unit(tag)
                ss=copy.deepcopy(current)
                ss.update(x=x,seed=seed)
                ss["pair"]=dict(ss["pair"],name=tag,background=source["pair"]["background"][:1000]+public,features=list(x))
                md=copy.deepcopy(metadata)
                if prefix=="R":
                    for contract in contracts:
                        md["columns"][contract["unit_column"]]="Public numeric transport encoding multiplier times_1, times_1000 or times_100."
                    md["columns"]["export_system"]="Public transport source identifier, not a target."
                md["columns"]={c:md["columns"].get(c,"Original public field: "+c) for c in x}
                config["field_metadata"][tag]=md
                reuse=code in CODES[:2] and seed==42
                if reuse:
                    original=PARENT/"units"/tag
                    old_source=joblib.load(original/"source.joblib")
                    for field in ("x","y","groups"):
                        assert old_source[field].equals(ss[field]),(tag,field)
                    assert read(original/"inner_splits.json")==folds
                    np.testing.assert_array_equal(np.load(original/"sealed_outer_y.npy"),source["y"].iloc[z])
                    assert joblib.load(original/"sealed_outer_X.joblib")["x"].equals(q)
                    assert parent_config["field_metadata"][tag]["columns"]==md["columns"]
                    shutil.copytree(original,unit,ignore=shutil.ignore_patterns("__pycache__"))
                    # Redirect copied transformation cache paths to this new root.
                    for path in unit.rglob("transform_state.joblib"):
                        value=joblib.load(path)
                        joblib.dump(remap(value),path)
                    for path in original.rglob("*"):
                        if path.is_file():protected[str(path.relative_to(ROOT))]=sha(path)
                    inherited.append(dict(tag=tag,source_inputs_and_prediction_equal=True))
                else:
                    unit.mkdir(parents=True)
                    joblib.dump(ss,unit/"source.joblib")
                    joblib.dump(dict(x=q,groups=source["groups"].iloc[z]),unit/"sealed_outer_X.joblib")
                    np.save(unit/"sealed_outer_y.npy",source["y"].iloc[z].to_numpy())
                    write(unit/"partition.json",dict(support=support.tolist(),remaining=rest.tolist(),role="development_only"))
                    write(unit/"inner_splits.json",folds)
                    write(unit/"public_export_contract.json",contracts)
                    write(unit/"format_schema.json",dict(categories=categories,numeric=numeric,evaluation="development_holdout_not_final_test"))
                    write(unit/"menu.json",registry(x,ss.get("date_columns",{})))
                    write(unit/"derived_ops.json",{})
                for name in ("source.joblib","sealed_outer_X.joblib","sealed_outer_y.npy","inner_splits.json","partition.json","public_export_contract.json","format_schema.json","menu.json"):
                    protected[str((unit/name).relative_to(ROOT))]=sha(unit/name)
                entries.append(dict(tag=tag,code=code,seed=seed,task="classification",train_rows=len(a),development_rows=len(z),pool_rows=len(source["x"]),source_rows=source_rows,
                    evaluation="historical_development_stress",outer_y_sha256=sha(unit/"sealed_outer_y.npy"),inherited_seed42=reuse))
            audits.append(dict(source=code,seed=seed,pool_rows=len(source["x"]),train_rows=len(a),assessment_rows=len(z),numeric=numeric,categories=categories,
                noncanonical_source_rows=float((train[[c["unit_column"] for c in contracts]]!="times_1").any(axis=1).mean()),
                new_missing_fraction=float((partial[numeric].isna()&source["x"][numeric].notna()).to_numpy().sum()/source["x"][numeric].notna().to_numpy().sum()),
                old_sealed_heldouts_opened=False))
    write(OUT/"run_config.json",config)
    baseline=read(ROOT/"results/e2_unified_independent_20261107/run_config.json")
    baseline.update(output_dir=str(OUT),field_metadata=config["field_metadata"],seeds=list(SEEDS),budgets=[5,10,20],lm_budget=20,unit_physical_limit=100)
    # Preserve final E2 resource repair while keeping native search parameters.
    baseline["native_timeouts"]=config["native_timeouts"]
    write(OUT/"baseline_config.json",baseline)
    write(OUT/"cases.json",entries)
    write(OUT/"data_audit.json",audits)
    write(OUT/"inherited_units.json",inherited)
    for name in ("cases.json","run_config.json","baseline_config.json","data_audit.json","inherited_units.json"):
        protected[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for path in (OUT/"runtime").rglob("*.py"):
        protected[str(path.relative_to(ROOT))]=sha(path)
    for relative,digest in read(PARENT/"protected_hashes_current.json").items():
        if relative.endswith((".ckpt",".safetensors")):protected[relative]=digest
    write(OUT/"protected_hashes.json",protected)
    frozen=dict(version["implementation"])
    for path in [Path(__file__).resolve(),CFG,ROOT/"scripts/e2_composite_corruption_pilot_20261118.py",ROOT/"designs/e2_composite_four_class_20261118/confirmed_experiment_protocol.json"]:
        frozen[str(path.relative_to(ROOT))]=sha(path)
    write(OUT/"implementation_frozen.json",frozen)
    b.verify()
    for row in entries:
        sc=baseline_scorer(row) if row["tag"].startswith("R_") else rt.scorer(row["tag"])
        f=sc.folds[0]
        a,z=f["train"],f["validation"]
        tr,va,yy=sc.transform(b.RAW,sc.x.iloc[a],sc.x.iloc[z],sc.y.iloc[a],sc.source["groups"].iloc[a],"zero_fit_preflight",sc.unit/"expanded_preflight")
        assert len(tr)==len(yy) and len(va)==len(z)
    write(OUT/"prepared.json",dict(time=time.time(),sources=4,seeds=list(SEEDS),units=24,inherited_units=4,new_fits=0,final_tests_opened=False))


def baseline_scorer(row):
    import numpy as np
    rt,_=pilot.bind()
    sc=rt.scorer(row["tag"])
    cfg=copy.deepcopy(read(OUT/"baseline_config.json"));cfg["seed"]=row["seed"]
    sc.c=cfg
    transform=sc.transform
    def apply(spec,*args,**kwargs):
        return transform(dict(spec,ids=spec.get("ids",[])),*args,**kwargs)
    sc.transform=apply
    original=sc.inner
    def inner(spec):
        if spec["kind"]!="ensemble":return original(spec)
        token=key(spec);p=sc.unit/"scores"/(token+".json")
        if p.exists():return read(p)
        from taskdq.e1 import metrics
        members=spec["members"]
        for member in members:inner(member)
        values=[]
        for i,f in enumerate(sc.folds):
            prediction=np.mean([np.load(sc.unit/"inner_models"/key(member)/str(i)/"predictions.npy") for member in members],axis=0)
            folder=sc.unit/"inner_models"/token/str(i);folder.mkdir(parents=True,exist_ok=True);np.save(folder/"predictions.npy",prediction)
            values.append(metrics(sc.y.iloc[f["validation"]],prediction,sc.task))
        scores={m:float(np.mean([v[m] for v in values])) for m in values[0]}
        record=dict(spec=spec,scores=scores,fold_scores=values,utility=scores["auc"])
        write(p,record);return record
    sc.inner=inner
    return sc


def full_worker(index,workers):
    pending=[r for r in records() if not (OUT/"units"/r["tag"]/"Full_complete.json").exists()]
    b.cases=lambda:pending
    b.full_worker(index,workers)


def baseline_worker(index,workers):
    pilot.bind()
    from taskdq.e2_unified import MethodEngine
    from taskdq.e2_pilot import run_method
    cfg=read(OUT/"baseline_config.json")
    engine=MethodEngine(cfg)
    rt,_=pilot.bind()
    generate=engine.generate
    def counted(messages,schema,unit,method,number):
        pilot.check_time()
        if not (unit/"methods"/method/"calls"/(str(number)+".json")).exists():
            rt.ledger("LM_ledger.json",dict(table=unit.name,method=method,number=number,status="started"),read(CFG)["limits"]["LM_generations"])
        return generate(messages,schema,unit,method,number)
    engine.generate=counted
    rows=[r for r in records() if r["tag"].startswith("R_")]
    for row in [r for i,r in enumerate(rows) if i%workers==index]:
        pilot.check_time();sc=baseline_scorer(row)
        for method in METHODS:
            folder=sc.unit/"methods"/method
            if (folder/"expanded_complete.json").exists():continue
            result=run_method(sc.c,sc,engine,method)
            if method not in LM_METHODS:
                for budget in (5,10,20):
                    write(folder/"checkpoints"/(f"{budget:03}.json"),dict(result,budget=budget,selected_before_outer_labels=True))
            for budget in (5,10,20):assert (folder/"checkpoints"/(f"{budget:03}.json")).exists()
            write(folder/"expanded_complete.json",dict(time=time.time(),method=method,status=result["status"]))
            print(row["tag"],method,"completed",flush=True)
        write(sc.unit/"baseline_complete.json",dict(time=time.time(),methods=list(METHODS)))
        b.verify()


def baseline_final_worker(index,workers):
    import joblib
    import numpy as np
    rt,_=pilot.bind()
    from taskdq.scoring import fit_predictor,predict
    rows=[r for r in records() if r["tag"].startswith("R_")]
    for row in [r for i,r in enumerate(rows) if i%workers==index]:
        sc=baseline_scorer(row);query=joblib.load(sc.unit/"sealed_outer_X.joblib")["x"]
        cache={}
        def final(spec):
            token=key(spec)
            if token in cache:return cache[token]
            folder=sc.unit/"baseline_final"/token
            if spec["kind"]=="ensemble":
                values=[final(member)[0] for member in spec["members"]]
                prediction=np.mean(values,axis=0);folder.mkdir(parents=True,exist_ok=True)
            elif (folder/"predictions.npy").exists():
                prediction=np.load(folder/"predictions.npy")
            else:
                pilot.check_time();folder.mkdir(parents=True,exist_ok=True)
                train,valid,y=sc.transform(spec,sc.x,query,sc.y,sc.source["groups"],"baseline_final",folder)
                n=rt.ledger("primary_ledger.json",dict(table=row["tag"],phase="baseline_final",status="started",folder=str(folder.relative_to(OUT))),read(CFG)["limits"]["primary_predictor_fits"])
                model=fit_predictor(train,y,sc.task,sc.c["model"]);prediction=predict(model,valid)
                joblib.dump(model,folder/"model.joblib");joblib.dump(dict(train=train,valid=valid,y_train=np.asarray(y)),folder/"data.joblib")
                np.testing.assert_array_equal(predict(joblib.load(folder/"model.joblib"),valid),prediction)
                rt.ledger_update("primary_ledger.json",n,status="completed")
            np.save(folder/"predictions.npy",prediction);cache[token]=(prediction,folder/"predictions.npy");return cache[token]
        inventory=[]
        for method in METHODS:
            for budget in (5,10,20):
                checkpoint=read(sc.unit/"methods"/method/"checkpoints"/(f"{budget:03}.json"))
                _,path=final(checkpoint["spec"])
                inventory.append(dict(source=row["code"],seed=row["seed"],tag=row["tag"],method=method,budget=budget,status=checkpoint["status"],spec=checkpoint["spec"],path=str(path.relative_to(OUT)),prediction_sha256=sha(path)))
        write(sc.unit/"baseline_endpoints_frozen.json",dict(time=time.time(),inventory=inventory))
    b.verify()


def report():
    import joblib
    import numpy as np
    import pandas as pd
    rt,_=pilot.bind()
    from taskdq.e1 import metrics
    from taskdq.scoring import predict
    b.verify();assert not (OUT/"assessment_scoring_receipt.json").exists()
    inventory=[];diagnostics=[]
    for row in records():
        unit=OUT/"units"/row["tag"]
        if row["tag"].startswith("R_"):
            inventory.extend(read(unit/"baseline_endpoints_frozen.json")["inventory"])
        else:
            endpoints=read(unit/"frozen_endpoints.json")["inventory"]
            chosen=next(e for e in endpoints if e["arm"]=="Full_neural_experts")
            for budget in (5,10,20):
                inventory.append(dict(source=row["code"],seed=row["seed"],tag=row["tag"],method="ours",budget=budget,status="passed",path=chosen["path"],prediction_sha256=chosen["prediction_sha256"],spec=chosen["spec"]))
            diagnostics.extend(dict(e,source=row["code"],seed=row["seed"]) for e in endpoints)
    assert len(inventory)==324
    for e in inventory+diagnostics:assert sha(OUT/e["path"])==e["prediction_sha256"]
    write(OUT/"all_endpoints_frozen.json",dict(time=time.time(),inventory=inventory,diagnostics=diagnostics,classification_tests_opened=False))
    ledger=read(OUT/"primary_ledger.json")
    for e in ledger:
        assert e["status"]=="completed"
        folder=OUT/e["folder"];data=joblib.load(folder/"data.joblib")
        np.testing.assert_array_equal(predict(joblib.load(folder/"model.joblib"),data["valid"]),np.load(folder/"predictions.npy"))
    write(OUT/"assessment_scoring_receipt.json",dict(time=time.time(),endpoints=324,scope="historical_development_stress",original_sealed_tests_loaded=False))
    rows=[]
    for e in inventory:
        target=np.load(OUT/"units"/e["tag"]/"sealed_outer_y.npy")
        score=metrics(target,np.load(OUT/e["path"]),dict(type="classification"))
        for metric,value in score.items():rows.append(dict(dataset=e["source"],seed=e["seed"],method=e["method"],budget=e["budget"],status=e["status"],metric=metric,value=value))
    diagnostics_rows=[]
    for e in diagnostics:
        target=np.load(OUT/"units"/e["tag"]/"sealed_outer_y.npy")
        score=metrics(target,np.load(OUT/e["path"]),dict(type="classification"))
        diagnostics_rows.append(dict(source=e["source"],seed=e["seed"],method=e["arm"],**score))
    dest=OUT/"deliverables";dest.mkdir()
    pd.DataFrame(rows).to_csv(dest/"all_seeds.csv",index=False,float_format="%.16g")
    pd.DataFrame(diagnostics_rows).to_csv(dest/"diagnostics.csv",index=False,float_format="%.16g")
    data=pd.DataFrame(rows)
    summary=data.groupby(["dataset","method","budget","metric"]).agg(mean=("value","mean"),sd=("value","std"),seeds=("seed","count"),fallback_seeds=("status",lambda v:int((v!="passed").sum()))).reset_index()
    summary.to_csv(dest/"summary.csv",index=False,float_format="%.16g")
    write(dest/"data_and_cost.json",dict(data=read(OUT/"data_audit.json"),inherited=read(OUT/"inherited_units.json"),new_primary_fits=len(ledger),LM_records=len(read(OUT/"LM_ledger.json")),query_records=len(read(OUT/"query_ledger.json")),expert_contexts=len(list((OUT/"units").glob("*/neural_CV/*/complete.json")))+len(list((OUT/"units").glob("*/neural_final/complete.json"))),scope="development_stress_only",all_signs_retained=True))
    write(OUT/"report_hashes.json",{p.name:sha(p) for p in dest.iterdir()})
    b.verify();pilot.check_time()
    write(OUT/"report_complete.json",dict(time=time.time(),rows=len(rows),endpoints=324))


def controller():
    import fcntl
    with (OUT/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/"dispatch.json").exists();assert (OUT/"prepared.json").exists();b.verify()
        lines=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()
        gpus=[int(v[0]) for line in lines if (v:=line.replace(" ","").split(",")) and int(v[0]) in [1,5,6,7] and int(v[1])<10000 and int(v[2])<5]
        assert gpus;write(OUT/"dispatch.json",dict(pid=os.getpid(),time=time.time(),deadline=pilot.deadline(),gpus=gpus))
        try:
            for stage in ("full_worker","neural_worker","final_worker","baseline_worker","baseline_final_worker"):
                pilot.phase(stage,gpus)
            report();write(OUT/"controller_complete.json",dict(time=time.time(),exit=0,wall_seconds=time.time()-read(OUT/"authorization_receipt.json")["start"]))
        except BaseException:
            write(OUT/"controller_failure.json",dict(time=time.time(),traceback=traceback.format_exc()));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage");p.add_argument("--index",type=int,default=0);p.add_argument("--workers",type=int,default=1);args=p.parse_args();configure()
    if args.stage in ("neural_worker","final_worker"):
        marker="neural_complete.json" if args.stage=="neural_worker" else "frozen_endpoints.json"
        pending=[r for r in normalized_records() if not (OUT/"units"/r["tag"]/marker).exists()]
        b.cases=lambda:pending;getattr(b,args.stage)(args.index,args.workers)
    elif args.stage.endswith("_worker"):
        globals()[args.stage](args.index,args.workers)
    else:globals()[args.stage]()
