import pickle

from sklearn import metrics

from utils import setup_logger, get_source_files, create_libs_set, fix_dataframe, LIB_OCCURRENCE_THRESHOLD, \
    TARGET_FIELD, MODEL_STORE_FILE

OUT_FILE_PATH = "validation.txt"


def calculate_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, str]:
    tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
    accuracy = metrics.accuracy_score(expected, predicted)
    precision = metrics.precision_score(expected, predicted, zero_division=0)
    recall = metrics.recall_score(expected, predicted, zero_division=0)
    f1 = metrics.f1_score(expected, predicted, zero_division=0)
    metrics_dict = {
        "True positive": tp,
        "False positive": fp,
        "False negative": fn,
        "True negative": tn,
        "Accuracy": "{:.4f}".format(accuracy),
        "Precision": "{:.4f}".format(precision),
        "Recall": "{:.4f}".format(recall),
        "F1": "{:.4f}".format(f1),
    }
    return metrics_dict


def write_metrics_to_file(predicted: list[bool], expected: list[bool], file_name: str) -> None:
    with open(file_name, 'w') as file:
        target_metrics = calculate_metrics(
            predicted,
            expected
        )
        for metric_name, metric_value in target_metrics.items():
            file.write(f"{metric_name}: {metric_value}\n")


def main():
    logger = setup_logger()
    train, validation, _ = get_source_files()
    set_libs = create_libs_set(train, LIB_OCCURRENCE_THRESHOLD)
    validation = fix_dataframe(validation, set_libs).rename(str, axis="columns")

    rfc = pickle.load(open(MODEL_STORE_FILE, 'rb'))
    cols_when_model_builds = rfc.feature_names_in_

    x_validation = validation[cols_when_model_builds]
    predicted = rfc.predict(x_validation)
    write_metrics_to_file(predicted, validation[TARGET_FIELD], OUT_FILE_PATH)
    logger.info(f"Result written to file {OUT_FILE_PATH}")


if __name__ == "__main__":
    main()
