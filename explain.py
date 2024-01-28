import typing

import pandas as pd


class FileHistoryEntry:
    def __init__(self, library: str, value: list[float], presence: bool):
        self.lib = library
        self.value = value
        self.presence = presence


class FileHistory:
    def __init__(self, entries: list[FileHistoryEntry]):
        self.entries = entries


class ExplainedDecisionTree:

    history: typing.ClassVar[list[FileHistory]] = []
    dataframe: typing.ClassVar[pd.DataFrame | None] = None

    def __init__(self, decision_tree: any, explain_tree_deep: int):
        self.decision_tree = decision_tree
        self.explain_tree_deep = explain_tree_deep  # max deep in tree for explaining

    def fit_explain_history(self, x_test: pd.DataFrame) -> None:
        """
        fill history property with path in decision tree (see FileHistory class)
        see https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        """
        history = []
        children_left = self.decision_tree.children_left
        children_right = self.decision_tree.children_right
        feature = self.decision_tree.feature
        threshold = self.decision_tree.threshold
        values = self.decision_tree.value
        for i in range(len(x_test)):
            current_node = 0
            current_item_history = []
            for _ in range(self.explain_tree_deep):
                feature_index = feature[current_node]
                feature_name = x_test.iloc[:, feature_index].name
                presence = x_test.iloc[:, feature_index][i] <= threshold[current_node]
                current_item_history.append(FileHistoryEntry(feature_name, values[current_node][0], presence))
                if children_left[current_node] != children_right[current_node]:  # not a leaf
                    if x_test.iloc[:, feature_index][i] <= threshold[current_node]:
                        current_node = children_left[current_node]
                    else:
                        current_node = children_right[current_node]
                else:
                    break
            history.append(FileHistory(current_item_history))
        self.history = history
        self.dataframe = x_test

    def get_comment_by_explain_history(self, predicted: list[bool]) -> list[str]:
        """
        fit_explain_history method should be called first
        transform data from history property (see fit_explain_history method) to human-readable comments
        """
        if self.dataframe is None:
            raise ValueError("ExplainedDecisionTree.fit_explain_history() should be called first")

        comments = []
        for i in range(len(self.dataframe)):
            current_entries = self.history[i].entries
            if len(current_entries) == 0:
                continue
            elif len(current_entries) == 1:
                # absence of single lab cannot be reason for treating file as virus
                if current_entries[0].value[1] > 0.4 and current_entries[0].presence:  # virus with single lib
                    comments.append(f"very dangerous library found -- {current_entries[0].lib}.")
                    continue
            comment_items = []
            for entry_index in range(1, len(current_entries)):
                current_entry = current_entries[entry_index]
                prev_entry = current_entries[entry_index - 1]
                if current_entry.value[1] - prev_entry.value[1] > 0.2:  # lib looks like virus
                    comment_items.append([current_entry.lib, prev_entry.lib])

            comment = ""
            if predicted[i]:
                if len(comment_items):
                    comment = f"dangerous: {', '.join(list(map(lambda x: f'({x[0]} with {x[1]})', comment_items)))}"
                elif current_entries[-1].value[1] < 0.8:
                    comment = "not sure that file is a virus"
                else:
                    comment = "cannot explain"
            comments.append(comment)
        return comments
