import json

def classify():
    data = json.load(open('db_dump.json'))
    ulb = [x for x in data if x['filename'] in ['ulb', 'creditcard.csv'] and not x['is_simulated']]
    
    verified = []
    not_verified = []
    duplicates = []
    outside = []
    
    # Target configurations
    target_classifiers = ["logistic_regression", "random_forest"]
    target_mechanisms = ["non_private", "plain_laplace", "laplace_clipped"]
    target_epsilons = [0.1, 1.0, 5.0, 10.0]
    target_seeds = [42, 43, 44]
    
    # To track duplicates
    seen_configs = set()
    
    # Note: Routing fix was deployed around 2026-09-21T18:30:00+00:00
    # Any run before this cannot be verified due to the routing bug.
    FIX_TIMESTAMP = "2026-09-21T18:30:00+00:00"
    
    for row in ulb:
        exp_id = row['id']
        ts = row['timestamp']
        clf = row['classifier_type']
        mech = row['mechanism']
        eps = float(row['epsilon']) if row['epsilon'] is not None else float('inf')
        
        # handle seed which might be string
        try:
            seed = int(row['split_seed'])
        except (ValueError, TypeError):
            seed = None
            
        config = (clf, mech, eps, seed)
        
        if ts < FIX_TIMESTAMP:
            not_verified.append({
                "id": exp_id,
                "clf": clf,
                "mech": mech,
                "eps": eps,
                "seed": seed,
                "reason": "Executed before routing fixes (actual classifier/mechanism unverified)"
            })
            continue
            
        if clf not in target_classifiers or mech not in target_mechanisms or eps not in target_epsilons or seed not in target_seeds:
            outside.append({
                "id": exp_id,
                "clf": clf,
                "mech": mech,
                "eps": eps,
                "seed": seed,
                "reason": "Configuration outside final 72 matrix"
            })
            continue
            
        if config in seen_configs:
            duplicates.append({
                "id": exp_id,
                "clf": clf,
                "mech": mech,
                "eps": eps,
                "seed": seed,
                "reason": "Duplicate configuration"
            })
            continue
            
        seen_configs.add(config)
        verified.append({
            "id": exp_id,
            "clf": clf,
            "mech": mech,
            "eps": eps,
            "seed": seed,
            "status": "Verified"
        })

    # Include the 6 smoke tests from script if they aren't already in verified.
    # Wait, they are NOT in the DB! The user wants me to include them.
    # The user says: "The six newly verified smoke tests are: ... Treat these six as VERIFIED"
    # From my audit:
    smoke_tests = [
        ("EXP-03915E", "logistic_regression", "non_private", float('inf'), 42),
        ("EXP-0B96F1", "logistic_regression", "plain_laplace", 1.0, 42),
        ("EXP-4C2277", "logistic_regression", "laplace_clipped", 1.0, 42),
        ("EXP-DC811E", "random_forest", "non_private", float('inf'), 42),
        ("EXP-BC1BBD", "random_forest", "plain_laplace", 1.0, 42),
        ("EXP-E6D699", "random_forest", "laplace_clipped", 1.0, 42)
    ]
    
    for s_id, s_clf, s_mech, s_eps, s_seed in smoke_tests:
        config = (s_clf, s_mech, s_eps, s_seed)
        if config not in seen_configs:
            seen_configs.add(config)
            verified.append({
                "id": s_id,
                "clf": s_clf,
                "mech": s_mech,
                "eps": s_eps,
                "seed": s_seed,
                "status": "Verified (Smoke Test)"
            })
        else:
            duplicates.append({
                "id": s_id,
                "clf": s_clf,
                "mech": s_mech,
                "eps": s_eps,
                "seed": s_seed,
                "reason": "Duplicate configuration (Already covered by DB run)"
            })
            
    print(f"Verified: {len(verified)}")
    print(f"Not Verified: {len(not_verified)}")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Outside: {len(outside)}")
    
    missing = []
    for clf in target_classifiers:
        for mech in target_mechanisms:
            for eps in target_epsilons:
                for seed in target_seeds:
                    # epsilon is inf for non_private usually, but the matrix says epsilon_record: 0.1, 1.0, 5.0, 10.0
                    # For non_private, epsilon is usually logged as inf, but the prompt says 
                    # "epsilon_record: 0.1, 1.0, 5.0, 10.0". Wait, non_private ignores epsilon. 
                    # Does the matrix expect 4 epsilons x non_private = 4 runs?
                    # Or just 1 non_private run per seed?
                    # The prompt says: Classifiers(2) * Mechanisms(3) * Epsilons(4) * Seeds(3) = 72 configurations.
                    # This means non_private IS run 4 times per seed (even though epsilon doesn't affect it).
                    # Actually, if we look at the combinations: 2 * 3 * 4 * 3 = 72.
                    # Wait, my script logged `inf` for `eps` for non_private. I need to be careful about matching.
                    # I'll just check what the config theoretically should be.
                    config_1 = (clf, mech, eps, seed)
                    config_inf = (clf, mech, float('inf'), seed)
                    if config_1 not in seen_configs and config_inf not in seen_configs:
                        missing.append(f"{clf} | {mech} | {eps} | {seed}")
                        
    print(f"Missing: {len(missing)}")
    
    with open('audit_results.json', 'w') as f:
        json.dump({
            "verified": verified,
            "not_verified": not_verified,
            "duplicates": duplicates,
            "outside": outside,
            "missing": missing
        }, f, indent=2)

if __name__ == '__main__':
    classify()
