import pickle

from explain import ExplainedDecisionTree
from utils import setup_logger, get_source_files, LIB_OCCURRENCE_THRESHOLD, create_libs_set, \
    fix_dataframe, MODEL_STORE_FILE

PREDICTED_FILE_NAME = "prediction.txt"
EXPLAIN_FILE_NAME = "explain.txt"
EXPLAIN_TREE_DEEP = 32


def write_prediction_to_file(predicted: list[bool], file_name: str) -> None:
    with open(file_name, 'w') as file:
        file.write("prediction\n")
        for predicted_state in predicted:
            file.write(str(int(predicted_state)) + "\n")


def write_comments_to_file(comments: list[str], file_name: str) -> None:
    with open(file_name, 'w') as file:
        for comment in comments:
            file.write(comment + "\n")


def main():
    logger = setup_logger()
    train, _, test = get_source_files()
    set_libs = create_libs_set(train, LIB_OCCURRENCE_THRESHOLD)
    test = fix_dataframe(test, set_libs).rename(str, axis="columns")

    rfc = pickle.load(open(MODEL_STORE_FILE, 'rb'))
    cols_when_model_builds = rfc.feature_names_in_
    test = test[cols_when_model_builds]

    predicted = rfc.predict(test)
    explained_dt = ExplainedDecisionTree(rfc.estimators_[0].tree_, EXPLAIN_TREE_DEEP)
    explained_dt.fit_explain_history(test)
    write_prediction_to_file(predicted, PREDICTED_FILE_NAME)
    write_comments_to_file(explained_dt.get_comment_by_explain_history(predicted), EXPLAIN_FILE_NAME)
    logger.info(f"Result written to files {PREDICTED_FILE_NAME} and {EXPLAIN_FILE_NAME}")


if __name__ == "__main__":
    main()
