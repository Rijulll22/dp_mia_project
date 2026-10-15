from backend.database import init_db, Experiment, SessionLocal
import json

init_db()
with SessionLocal() as db:
    records = db.query(Experiment).filter(Experiment.is_simulated == False).order_by(Experiment.timestamp.desc()).limit(8).all()
    for r in records:
        metrics = json.loads(r.metrics) if r.metrics else {}
        print('-'*40)
        print(f'ID: {r.id}, Timestamp: {r.timestamp}')
        print(f'Model: {r.classifier_type}, Mechanism: {r.mechanism}')
        print(f'Epsilon Record: {r.epsilon_record}, Epsilon Feature: {r.epsilon_feature}, Feature Count: {r.feature_count}')
        print(f'Seed: {r.split_seed}, Composition Method: {r.composition_method}')
        print(f'Utility => Acc: {metrics.get("accuracy")}, Prec: {metrics.get("precision")}, Rec: {metrics.get("recall")}, F1: {metrics.get("f1")}')
        print(f'         => ROC-AUC: {metrics.get("rocAuc")}, PR-AUC: {metrics.get("prAuc")}')
        print(f'Fraud   => Prec: {metrics.get("fraud_precision")}, Rec: {metrics.get("fraud_recall")}, F1: {metrics.get("fraud_f1")}')
        print(f'MIA     => Acc: {r.mia_accuracy}, AUC: {r.mia_auc}, Adv: {r.attacker_advantage}')
        print(f'TPR     => @1% FPR: {r.tpr_at_1fpr}, @0.1% FPR: {r.tpr_at_01fpr}')
