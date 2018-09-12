#!/usr/bin/env python
"""
Use a random forest to predict classes.

:Authors:
    Jacob Porter <jsporter@vt.edu>
"""

import sys
import datetime
import argparse
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


# The filename to use to store the model.
MODEL_NAME = "model.pck"
# The filename to use to store the LabelEncoder.
ENCODER_NAME = "encoder.pck"
# Feature importances or coefficients.
FEATURE_COEFF = "feature_imp.txt"
# The minimum amount to use in a Bayesian optmization search space.
MIN_SEARCH = 2e-7
# The acquisition function to use in Bayesian optimization.
ACQUISITION_FUNCTION = "EI"
# Classifier shorthand.
CLASSIFIER_CHOICES = {"rf": 0,
                      "lr": 1,
                      "ab": 0}


def getPipeRF(num_features):
    """
    Return a pipeline and a search space for a random forest.

    Parameters
    ----------
    num_features: int
        The number of features that the classifier will be trained on.
        This determines the max_depth  in the search space.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing a random forest classifier followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    rf = RandomForestClassifier()
    search_space = {'rf__max_depth': Integer(2, num_features),
                    'rf__max_features': Real(MIN_SEARCH,
                                             1.0, prior='log-uniform'),
                    'rf__criterion': Categorical(['gini', 'entropy']),
                    'rf__min_samples_split': Real(MIN_SEARCH, 1.0,
                                                  prior='log-uniform'),
                    'rf__min_samples_leaf': Real(MIN_SEARCH, 0.5,
                                                 prior='log-uniform')
                    }
    return Pipeline([('rf', rf)]), search_space


def getPipeLR(num_features):
    """
    Return a pipeline and a search space for a random forest.

    Parameters
    ----------
    num_features: int
        The number of features that the classifier will be trained on.
        This determines the max_depth  in the search space.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing a random forest classifier followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    lr = LogisticRegression(solver='sag')
    search_space = {'lr__C': Real(MIN_SEARCH, 1.0, prior="uniform"),
                    'lr__multi_class': Categorical(['ovr', 'multinomial'])
                    }
    return Pipeline([('ss', StandardScaler()), ('lr', lr)]), search_space


def getPipeAB(num_features):
    """
    Return a pipeline and a search space for a random forest.

    Parameters
    ----------
    num_features: int
        The number of features that the classifier will be trained on.
        This determines the max_depth  in the search space.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing a random forest classifier followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    ab = AdaBoostClassifier()
    search_space = {'ab__n_estimators': Integer(32, 256),
                    }
    return Pipeline([('ss', StandardScaler()), ('ab', ab)]), search_space


def train(features, classes, model_path, classifier='RandomForest',
          iterations=25, folds=3, processes=3, verbose=0):
    """
    Train the model.

    Parameters
    ----------
    features: array-like or numpy array
        Array of features.
    classes: array-like
        A list or array of class labels.  These can be strings.
    model_path: str
        The directory to save the model and encoder.
    iterations: int
        The amount of hyperparameter searching to do.
    folds: int
        The number of cross validation folds to perform.
    processes: int
        The number of processes to use to do fitting.
        In Bayesian optimization, there won't be an efficiency improvement if
        this is larger than the number of folds.
    verbose: int
        The verbosity of the classifier.  A higher number makes the classifier
        more verbose.

    Returns
    -------
    Classifier, LabelEncoder
        The best classifier according the cross-validation followed by the
        trained LabelEncoder.

    """
    if classifier == 'RandomForest':
        getPipeFunc = getPipeRF
    elif classifier == 'LogisticRegression':
        getPipeFunc = getPipeLR
    elif classifier == 'AdaBoost':
        getPipeFunc = getPipeAB
    classifier_pipe, search_space = getPipeFunc(features.shape[1])
    encoder = LabelEncoder()
    encoder.fit(classes)
    scoring = make_scorer(accuracy_score)
    optimizer_kwargs = {'acq_func': ACQUISITION_FUNCTION}
    cv = BayesSearchCV(classifier_pipe, search_space, cv=folds,
                       verbose=verbose, n_iter=iterations, n_jobs=processes,
                       optimizer_kwargs=optimizer_kwargs,
                       scoring=scoring)
    cv.fit(features, encoder.transform(classes))
    print("The best score: {}".format(cv.best_score_), file=sys.stderr)
    print("The best parameters: {}".format(cv.best_params_), file=sys.stderr)
    return cv.best_estimator_, encoder


def __get_classes(model):
    for classifier in CLASSIFIER_CHOICES:
        if classifier in model.named_steps:
            return model.named_steps[classifier].classes_


def predict(model, encoder, features):
    """
    Predict classes.

    Parameters
    ----------
    model: Classifier
        A sklearn machine learning model to do prediction.
    encoder: LabelEncoder
        A trained label encoder.
    features: numpy array
        A numpy array of features.

    Returns
    -------
    iterable, iterable, iterable
        The predicted labels, the predicted probabilities, and the
        order of class labels for the probabilities

    """
    y_predict = model.predict(features)
    labels = encoder.inverse_transform(y_predict)
    proba = model.predict_proba(features)
    order = encoder.inverse_transform(__get_classes(model))
    return labels, proba, order


def evaluate(model, encoder, features, classes):
    """
    Give reports on features, predictive performance, etc from a prediction.

    Parameters
    ----------
    model: Classifier
        A sklearn machine learning model.
    encoder: LabelEncoder
        A label encoder to use on the classes.
    features: numpy array
        A numpy array representing the features.
    classes: list
        A list of classes corresponding to the features.

    Returns
    -------
    dict
        A dictionary representing model evaluation reports.

    """
    y_predict, _, _ = predict(model, encoder, features)
    labels = list(encoder.classes_)
    my_confusion_matrix = confusion_matrix(classes, y_predict, labels=labels)
    my_report = classification_report(classes, y_predict, target_names=labels,
                                      digits=8)
    acc = accuracy_score(classes, y_predict)
    f1 = f1_score(classes, y_predict, labels=labels, average='weighted')
    return {"acc": acc, "f1": f1, "confusion_matrix": my_confusion_matrix,
            "classification_report": my_report}


def get_features_classes(feature_path, class_path):
    """
    Get the feature array and class list from files.

    Parameters
    ----------
    feature_path: str
        The location of the features file.
    class_path: str
        The location of the class file/

    Returns
    -------
    array, list<str>
        A numpy array of features and a list of classes.

    """
    features = []
    classes = []
    if feature_path:
        with open(feature_path) as feature_file:
            for line in feature_file:
                observation = list(map(float, line.strip().split()))
                features.append(observation)
    if class_path:
        with open(class_path) as class_file:
            classes = [line.strip() for line in class_file]
    return np.asarray(features), classes


def load_model(model_path):
    """
    Load the machine learning model and label encoder.

    Parameters
    ----------
    model_path: str
        The path to the directory where the model and label encoder are stored.

    Returns
    -------
    model, encoder
        The model and the label encoder.

    """
    model = pickle.load(open(os.path.join(model_path, MODEL_NAME), 'rb'))
    encoder = pickle.load(open(os.path.join(model_path, ENCODER_NAME), 'rb'))
    return model, encoder


def __get_feature_imp(model):
    for classifier in CLASSIFIER_CHOICES:
        if classifier in model.named_steps:
            if CLASSIFIER_CHOICES[classifier] == 0:
                return model.named_steps[classifier].feature_importances_
            else:
                return model.named_steps[classifier].coef_


def save_model(model, encoder, model_path):
    """
    Save the model and label encoder.

    Parameters
    ----------
    model: sklearn ml model
        A model to save.
    encoder: LabelEncoder
        A label encoder to save.
    model_path: str
        A directory to save the model and encoder to.

    Returns
    -------
    None

    """
    pickle.dump(model, open(os.path.join(model_path,
                                         MODEL_NAME), 'wb'))
    pickle.dump(encoder, open(os.path.join(model_path, ENCODER_NAME), 'wb'))
    print(str(__get_feature_imp(model)), file=open(
        os.path.join(model_path, FEATURE_COEFF), 'w'))


class ArgClass:
    """
    So that I don't have to duplicate argument info when
    the same thing is used in more than one mode.
    """

    def __init__(self, *args, **kwargs):
        """Store arguments."""
        self.args = args
        self.kwargs = kwargs


def main():
    """Parse arguments."""
    tick = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=("Train, predict, and "
                                                  "evaluate a random forest "
                                                  "model."),
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    classifier = ArgClass("-c", "--classifier", type=str,
                          help=("RandomForest, LogisticRegression, or "
                                "AdaBoost."),
                          default='RandomForest')
    iterations = ArgClass("-i", "--iterations", type=int,
                          help="Iterations of Bayesian optimization to do.",
                          default=25)
    folds = ArgClass("-f", "--folds", type=int,
                     help="The number of folds of cross validation to do.",
                     default=3)
    processes = ArgClass("-p", "--processes", type=int,
                         help="The number of processes to do training on.",
                         default=3)
    verbose = ArgClass("-v", "--verbose", type=int,
                       help="Controls the verbosity of the classifier.",
                       default=2)
    subparsers = parser.add_subparsers(help="sub-commands", dest="mode")
    parser_train = subparsers.add_parser("train", help="Train the model.",
                                         formatter_class=argparse.
                                         ArgumentDefaultsHelpFormatter)
    parser_train.add_argument("model_path", type=str,
                              help=("The location where the "
                                    "model is stored. "))
    parser_train.add_argument("feature_path", type=str,
                              help=("The location of the features data."))
    parser_train.add_argument("class_path", type=str,
                              help=("The location of the class data."))
    parser_train.add_argument(*classifier.args, **classifier.kwargs)
    parser_train.add_argument(*iterations.args, **iterations.kwargs)
    parser_train.add_argument(*folds.args, **folds.kwargs)
    parser_train.add_argument(*processes.args, **processes.kwargs)
    parser_train.add_argument(*verbose.args, **verbose.kwargs)
    parser_predict = subparsers.add_parser("predict",
                                           help=("Make predictions "
                                                 "given a saved model. "
                                                 "The predictions are "
                                                 "written to stdout."),
                                           formatter_class=argparse.
                                           ArgumentDefaultsHelpFormatter)
    parser_predict.add_argument("model_path", type=str,
                                help=("The location where the "
                                      "model is stored. "))
    parser_predict.add_argument("feature_path", type=str,
                                help=("The location of the features data."))
    parser_evaluate = subparsers.add_parser("evaluate",
                                            help=("Make predictions and "
                                                  "evaluate their predictive "
                                                  "performance.  "
                                                  "The evaluation is "
                                                  "written to stdout. "),
                                            formatter_class=argparse.
                                            ArgumentDefaultsHelpFormatter)
    parser_evaluate.add_argument("model_path", type=str,
                                 help=("The location where the "
                                       "model is stored. "))
    parser_evaluate.add_argument("feature_path", type=str,
                                 help=("The location of the features data."))
    parser_evaluate.add_argument("class_path", type=str,
                                 help=("The location of the class data."))
    args = parser.parse_args()
    print(args, file=sys.stderr)
    mode = args.mode
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if mode == "train":
        features, classes = get_features_classes(args.feature_path,
                                                 args.class_path)
        estimator, encoder = train(features, classes, args.model_path,
                                   classifier=args.classifier,
                                   iterations=args.iterations,
                                   folds=args.folds, processes=args.processes,
                                   verbose=args.verbose)
        save_model(estimator, encoder, args.model_path)
    elif mode == "predict":
        features, _ = get_features_classes(args.feature_path, None)
        model, encoder = load_model(args.model_path)
        labels, proba, order = predict(model, encoder, features)
        print("{}\t{}".format("predicted_class", "\t".join(order)))
        for label, p in zip(labels, proba):
            print("{}\t{}".format(label,
                                  str(p).replace("[", "").replace("]", "")))
            # print("{}\t{}".format(str(label), "\t".join(str(p))))
    elif mode == "evaluate":
        features, classes = get_features_classes(args.feature_path,
                                                 args.class_path)
        model, encoder = load_model(args.model_path)
        evaluation_reports = evaluate(model, encoder, features, classes)
        for report in evaluation_reports:
            print(report)
            print(evaluation_reports[report])
            print("")
    else:
        parser.error("No such command found: {}".format(mode))
    tock = datetime.datetime.now()
    print("The process took time: {}".format(tock - tick), file=sys.stderr)


if __name__ == "__main__":
    main()
