import pickle

from sklearn.ensemble import RandomForestClassifier

from utils import setup_logger, get_source_files, LIB_OCCURRENCE_THRESHOLD, create_libs_set, fix_dataframe, \
    calculate_classes_balance, TARGET_FIELD, MODEL_STORE_FILE


def main():
    logger = setup_logger()
    train, _, _ = get_source_files()
    set_libs = create_libs_set(train, LIB_OCCURRENCE_THRESHOLD)
    train = fix_dataframe(train, set_libs).rename(str, axis="columns")
    with_virus_cnt, without_virus_cnt = calculate_classes_balance(train)
    logger.info(f"With virus: {with_virus_cnt}, without virus: {without_virus_cnt}")
    logger.info(f"Training on {len(set_libs)} unique libs")
    rfc = RandomForestClassifier(n_estimators=25, criterion='entropy', random_state=17, class_weight='balanced')
    print(train.drop([TARGET_FIELD, "filename"], axis=True).shape)
    rfc.fit(train.drop([TARGET_FIELD, "filename"], axis=True), train[TARGET_FIELD])  # slow operation
    pickle.dump(rfc, open(MODEL_STORE_FILE, 'wb'))


if __name__ == "__main__":
    main()
