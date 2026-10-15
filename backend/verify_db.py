import json
from database import init_db, Experiment, SessionLocal
init_db()
with SessionLocal() as db:
    records = db.query(Experiment).filter(
        Experiment.is_simulated == False,
        Experiment.epsilon_feature != None
    ).order_by(Experiment.timestamp.desc()).all()
    print(f'New experiments with correct epsilon accounting: {len(records)}')
    for r in records[:5]:
        m = json.loads(r.metrics) if r.metrics else {}
        f1 = m.get("f1")
        pr = m.get("prAuc")
        ff = m.get("fraud_f1")
        print(f'{r.id}: eps={r.epsilon}, eps_feature={r.epsilon_feature}, d={r.feature_count}')
        print(f'  F1={f1}, PR-AUC={pr}, fraud_F1={ff}')
        print(f'  MIA-acc={r.mia_accuracy}, MIA-AUC={r.mia_auc}, Advantage={r.attacker_advantage}')
        print(f'  TPR_1pct={r.tpr_at_1fpr}, composition={r.composition_method}')
        if r.epsilon_feature and r.feature_count:
            check = round(r.epsilon_feature * r.feature_count, 6)
            print(f'  VERIFY: {r.epsilon_feature} x {r.feature_count} = {check} vs record={r.epsilon_record}')
        print()

