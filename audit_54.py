import json

def audit():
    data = json.load(open('db_dump.json'))
    ulb = [x for x in data if x['filename'] in ['ulb', 'creditcard.csv'] and not x['is_simulated']]

    # Target values
    target_classifiers = ["logistic_regression", "random_forest"]
    target_dp_mechanisms = ["plain_laplace", "laplace_clipped"]
    target_epsilons = [0.1, 1.0, 5.0, 10.0]
    target_seeds = [42, 43, 44]

    FIX_TIMESTAMP = "2026-09-21T18:30:00+00:00"

    # Store verified ones by their logical tuple
    # Non-private: (clf, "non_private", None, seed)
    # DP: (clf, mech, eps, seed)
    verified_configs = {}
    duplicates = []
    outside = []
    not_verified = []

    for row in ulb:
        exp_id = row['id']
        ts = row['timestamp']
        clf = row['classifier_type']
        mech = row['mechanism']
        eps = float(row['epsilon']) if row['epsilon'] is not None else float('inf')

        try:
            seed = int(row['split_seed'])
        except (ValueError, TypeError):
            seed = None

        if ts < FIX_TIMESTAMP:
            not_verified.append(exp_id)
            continue

        if clf not in target_classifiers or seed not in target_seeds:
            outside.append(exp_id)
            continue

        if mech == "non_private":
            config = (clf, "non_private", None, seed)
        else:
            if mech not in target_dp_mechanisms or eps not in target_epsilons:
                outside.append(exp_id)
                continue
            config = (clf, mech, eps, seed)

        if config in verified_configs:
            duplicates.append(exp_id)
        else:
            verified_configs[config] = exp_id

    # Include smoke tests directly
    smoke_tests = [
        ("EXP-03915E", "logistic_regression", "non_private", None, 42),
        ("EXP-0B96F1", "logistic_regression", "plain_laplace", 1.0, 42),
        ("EXP-4C2277", "logistic_regression", "laplace_clipped", 1.0, 42),
        ("EXP-DC811E", "random_forest", "non_private", None, 42),
        ("EXP-BC1BBD", "random_forest", "plain_laplace", 1.0, 42),
        ("EXP-E6D699", "random_forest", "laplace_clipped", 1.0, 42)
    ]

    for exp_id, clf, mech, eps, seed in smoke_tests:
        config = (clf, mech, eps, seed)
        if config in verified_configs:
            # We don't overwrite the smoke test, we treat it as the verified one,
            # and the existing DB one as a duplicate
            duplicates.append(verified_configs[config])
            verified_configs[config] = exp_id
        else:
            verified_configs[config] = exp_id

    # Output verified
    print("### NON-PRIVATE")
    for (clf, mech, eps, seed), exp_id in sorted(verified_configs.items()):
        if mech == "non_private":
            print(f"{clf} | {seed} | {exp_id}")

    print("\n### DP")
    for (clf, mech, eps, seed), exp_id in sorted(verified_configs.items()):
        if mech != "non_private":
            print(f"{clf} | {mech} | {eps} | {seed} | {exp_id}")

    print(f"\nTOTAL VERIFIED CONFIGURATIONS: {len(verified_configs)}")

    # Missing
    print("\n### MISSING")
    missing_count = 0
    # non-private missing
    for clf in target_classifiers:
        for seed in target_seeds:
            config = (clf, "non_private", None, seed)
            if config not in verified_configs:
                print(f"{clf} | non_private | {seed}")
                missing_count += 1

    # dp missing
    for clf in target_classifiers:
        for mech in target_dp_mechanisms:
            for eps in target_epsilons:
                for seed in target_seeds:
                    config = (clf, mech, eps, seed)
                    if config not in verified_configs:
                        print(f"{clf} | {mech} | {eps} | {seed}")
                        missing_count += 1

    print(f"\nTOTAL MISSING CONFIGURATIONS: {missing_count}")

if __name__ == '__main__':
    audit()
