#!/usr/bin/env python
"""
Train an estimator model to make predictions given features and responses.

:Authors:
    Jacob Porter <jsporter@vt.edu>
"""

import argparse
import datetime
import os
import pickle
import sys

from numpy import asarray
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, LogisticRegression,
                                  RidgeClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, explained_variance_score,
                             f1_score, make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# The filename to use to store the model.
MODEL_NAME = "model.pck"
# The filename to use to store the LabelEncoder.
ENCODER_NAME = "encoder.pck"
# Feature importances or coefficients.
FEATURE_COEFF = "feature_imp_or_coeff.txt"
# The name of the estimator.
NAME_PATH = "name.txt"
# The minimum amount to use in a Bayesian optmization search space.
MIN_SEARCH = 2e-12
# The acquisition function to use in Bayesian optimization.
ACQUISITION_FUNCTION = "EI"
# The number of random forest estimators to use.
N_ESTIMATORS = 15
# Controls whether the feature input has a header.
HEADER_EXISTS = False


def getPipeRFC(num_features, n_estimators=N_ESTIMATORS):
    """
    Return a pipeline and a search space for a random forest classifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real, Categorical, Integer
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None)
    search_space = {
        'randomforestclassifier__n_estimators':
        Integer(8, 15),
        'randomforestclassifier__max_features':
        Real(MIN_SEARCH, 1.0, prior='uniform'),
        'randomforestclassifier__criterion':
        Categorical(['gini', 'entropy']),
        'randomforestclassifier__min_samples_split':
        Real(MIN_SEARCH, 1.0, prior='log-uniform'),
        'randomforestclassifier__min_samples_leaf':
        Real(MIN_SEARCH, 0.5, prior='log-uniform'),
        # 'randomforestclassifier__max_depth':
        # Integer(2, num_features),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('randomforestclassifier', rf)]), search_space)


def getPipeDTC(num_features):
    """
    Return a pipeline and a search space for a decision tree classifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real, Categorical, Integer
    dt = DecisionTreeClassifier()
    search_space = {
        'decisiontreeclassifier__max_features':
        Real(MIN_SEARCH, 1.0, prior='uniform'),
        'decisiontreeclassifier__criterion':
        Categorical(['gini', 'entropy']),
        'decisiontreeclassifier__min_samples_split':
        Real(MIN_SEARCH, 1.0, prior='log-uniform'),
        'decisiontreeclassifier__min_samples_leaf':
        Real(MIN_SEARCH, 0.5, prior='log-uniform'),
        'decisiontreeclassifier__max_depth':
        Integer(2, num_features),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('decisiontreeclassifier', dt)]), search_space)


def getPipeLR(num_features):
    """
    Return a pipeline and a search space for a logistic regression classifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real, Categorical
    lr = LogisticRegression(solver='sag')
    search_space = {
        'logisticregression__C': Real(MIN_SEARCH, 1.0, prior="uniform"),
        'logisticregression__multi_class': Categorical(['ovr', 'multinomial'])
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('logisticregression', lr)]), search_space)


def getPipeABF(num_features):
    """
    Return a pipeline and a search space for an AdaBoostClassifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Integer, Real
    # Using a more complex random forest classifier hurt performance on a
    # sample data set.  Perhaps some data would benefit from more complexity.
    ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(
        n_estimators=5,
        criterion='entropy',
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
    ),
                            algorithm='SAMME.R')
    search_space = {
        'adaboostforest__n_estimators': Integer(1, 32),
        'adaboostforest__learning_rate': Real(MIN_SEARCH, 1.0,
                                              prior='uniform'),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('adaboostforest', ab)]), search_space)


def getPipeABT(num_features):
    """
    Return a pipeline and a search space for an AdaBoostClassifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Integer, Real
    # Using a more complex decision tree classifier hurt performance on a
    # sample data set.  Perhaps some data would benefit from more complexity.
    ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                            algorithm='SAMME.R')
    search_space = {
        'adaboosttree__n_estimators': Integer(5, 150),
        'adaboosttree__learning_rate': Real(MIN_SEARCH, 1.0, prior='uniform'),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('adaboosttree', ab)]), search_space)


def getPipeGB(num_features):
    """
    Return a pipeline and a search space for a GradientBoostingClassifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Integer, Real, Categorical
    gb = GradientBoostingClassifier(loss='deviance')
    search_space = {
        'gradboost__n_estimators': Integer(50, 150),
        'gradboost__max_features': Real(MIN_SEARCH, 1.0, prior='uniform'),
        'gradboost__criterion': Categorical(['friedman_mse',
                                             'mse']),  # mae is very slow.
        'gradboost__min_samples_split': Real(MIN_SEARCH,
                                             1.0,
                                             prior='log-uniform'),
        'gradboost__min_samples_leaf': Real(MIN_SEARCH,
                                            0.5,
                                            prior='log-uniform'),
        'gradboost__max_depth': Integer(2, num_features),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('gradboost', gb)]), search_space)


def getPipeRC(num_features):
    """
    Return a pipeline and a search space for a RidgeClassifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real
    rc = RidgeClassifier(solver='auto')
    search_space = {
        'ridgeclassifier__alpha': Real(MIN_SEARCH, 1.0, prior="uniform")
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('ridgeclassifier', rc)]), search_space)


def getPipeLSVC(num_features):
    """
    Return a pipeline and a search space for a linear support vector
    classifier.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real
    lsvc = LinearSVC()
    search_space = {
        'supportvectorclassifier__C': Real(MIN_SEARCH, 1.0, prior="uniform")
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('supportvectorclassifier', lsvc)]), search_space)


def getPipeLSVR(num_features):
    """
    Return a pipeline and a search space for a linear support vector
    regressor.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real
    lsvr = LinearSVR()
    search_space = {
        'supportvectorregressor__C': Real(MIN_SEARCH, 1.0, prior="uniform")
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('supportvectorregressor', lsvr)]), search_space)


def getPipeEN(num_features):
    """
    Return a pipeline and a search space for an elastic net regressor.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real
    en = ElasticNet()
    search_space = {
        'elasticnet__alpha': Real(MIN_SEARCH, 1.0, prior="uniform"),
        'elasticnet__l1_ratio': Real(MIN_SEARCH, 1.0, prior="uniform")
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('elasticnet', en)]), search_space)


def getPipeRFR(num_features, n_estimators=N_ESTIMATORS):
    """
    Return a pipeline and a search space for a random forest regressor.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real, Categorical, Integer
    rfr = RandomForestRegressor(n_estimators=n_estimators)
    search_space = {
        'randomforestregressor__n_estimators':
        Integer(8, 15),
        'randomforestregressor__max_features':
        Real(MIN_SEARCH, 1.0, prior='uniform'),
        'randomforestregressor__criterion':
        Categorical(['mse', 'mae']),
        'randomforestregressor__min_samples_split':
        Real(MIN_SEARCH, 1.0, prior='log-uniform'),
        'randomforestregressor__min_samples_leaf':
        Real(MIN_SEARCH, 0.5, prior='log-uniform'),
        # 'randomforestregressor__max_depth':
        # Integer(2, num_features),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('randomforestregressor', rfr)]), search_space)


def getPipeDTR(num_features):
    """
    Return a pipeline and a search space for a decision tree regressor.

    Parameters
    ----------
    num_features: int
        The number of features that the estimator will be trained on.

    Returns
    -------
    Pipeline, dict
        A pipeline object representing an estimator followed
        by a dictionary representing a search space for Bayesian
        hyperparameter optimization.

    """
    from skopt.space import Real, Categorical, Integer
    dtr = DecisionTreeRegressor()
    search_space = {
        'decisiontreeregressor__max_features':
        Real(MIN_SEARCH, 1.0, prior='uniform'),
        'decisiontreeregressor__criterion':
        Categorical(['friedman_mse', 'mse', 'mae']),
        'decisiontreeregressor__min_samples_split':
        Real(MIN_SEARCH, 1.0, prior='log-uniform'),
        'decisiontreeregressor__min_samples_leaf':
        Real(MIN_SEARCH, 0.5, prior='log-uniform'),
        'decisiontreeregressor__max_depth':
        Integer(2, num_features),
    }
    return (Pipeline([('ss', StandardScaler()),
                      ('decisiontreeregressor', dtr)]), search_space)


# The estimator choices and corresponding pipelines.
# For position 1, a 0 indicates the classifier has a
# feature_importances_ attribute.
# For position 1, a 1 indicates the classifier has a coef_ attribute.
# Position 2 is the more readable name of the estimator.
# Position 3 is a 0 for a classifier and a 1 for a regressor.
ESTIMATOR_CHOICES = {
    "randomforestclassifier": (getPipeRFC, 0, 'RandomForestClassifier', 0),
    "decisiontreeclassifier": (getPipeDTC, 0, 'DecisionTreeClassifier', 0),
    "logisticregression": (getPipeLR, 1, 'LogisticRegression', 0),
    "ridgeclassifier": (getPipeRC, 1, 'RidgeClassifier', 0),
    "adaboostforest": (getPipeABF, 0, 'AdaBoostForest', 0),
    "adaboosttree": (getPipeABT, 0, 'AdaBoostTree', 0),
    "gradboost": (getPipeGB, 0, 'GradBoost', 0),
    "supportvectorclassifier": (getPipeLSVC, 1, 'SupportVectorClassifier', 0),
    "supportvectorregressor": (getPipeLSVR, 1, 'SupportVectorRegressor', 1),
    "elasticnet": (getPipeEN, 1, 'ElasticNet', 1),
    "randomforestregressor": (getPipeRFR, 0, 'RandomForestRegressor', 1),
    "decisiontreeregressor": (getPipeDTR, 0, 'DecisionTreeRegressor', 1)
}


def train(features,
          responses,
          model_path,
          estimator='RandomForestClassifier',
          iterations=25,
          folds=3,
          processes=3,
          verbose=0):
    """
    Train the model.

    Parameters
    ----------
    features: array-like or numpy array
        Array of features.
    responses: array-like
        A list or array of responses.
        For classification, these can be strings.
    model_path: str
        The directory to save the model and encoder.
    estimator: str
        The estimator to use.
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
    from skopt import BayesSearchCV
    estimator = estimator.lower()
    getPipeFunc = ESTIMATOR_CHOICES[estimator][0]
    estimator_pipe, search_space = getPipeFunc(features.shape[1])
    is_regressor = ESTIMATOR_CHOICES[estimator][3]
    if not is_regressor:
        encoder = LabelEncoder()
        encoder.fit(responses)
        scoring = make_scorer(accuracy_score)
    else:
        responses = [float(response) for response in responses]
        responses = asarray(responses)
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
    optimizer_kwargs = {'acq_func': ACQUISITION_FUNCTION}
    cv = BayesSearchCV(estimator_pipe,
                       search_space,
                       cv=folds,
                       verbose=verbose,
                       n_iter=iterations,
                       n_jobs=processes,
                       optimizer_kwargs=optimizer_kwargs,
                       scoring=scoring)
    if not is_regressor:
        cv.fit(features, encoder.transform(responses))
    else:
        cv.fit(features, responses)
    print("The best score: {}".format(cv.best_score_), file=sys.stderr)
    print("The best parameters: {}".format(cv.best_params_), file=sys.stderr)
    sys.stderr.flush()
    if not is_regressor:
        return cv.best_estimator_, encoder
    else:
        return cv.best_estimator_, None


def __get_classes(model, name=None):
    if name:
        return model.named_steps[name].classes_
    else:
        for estimator in ESTIMATOR_CHOICES:
            if estimator in model.named_steps:
                return model.named_steps[estimator].classes_


def predict(model, name, features, encoder=None):
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
    name = name.lower()
    is_regressor = ESTIMATOR_CHOICES[name][3]
    y_predict = model.predict(features)
    if not is_regressor:
        labels = encoder.inverse_transform(y_predict)
        try:
            proba = model.predict_proba(features)
        except AttributeError:
            proba = None
        order = encoder.inverse_transform(__get_classes(model, name))
        return labels, proba, order
    else:
        return y_predict, None, None


def evaluate(model, name, features, responses, encoder=None):
    """
    Give a report on predictive performance from predicting from the features.

    Parameters
    ----------
    model: Classifier
        A sklearn machine learning model.
    encoder: LabelEncoder
        A label encoder to use on the classes.
    features: numpy array
        A numpy array representing the features.
    responses: list
        A list of responses.

    Returns
    -------
    dict
        A dictionary representing model evaluation reports.

    """
    name = name.lower()
    is_regressor = ESTIMATOR_CHOICES[name][3]
    y_predict, _, _ = predict(model, name, features, encoder)
    if not is_regressor:
        labels = list(encoder.classes_)
        my_confusion_matrix = confusion_matrix(responses,
                                               y_predict,
                                               labels=labels)
        my_report = classification_report(responses,
                                          y_predict,
                                          target_names=labels,
                                          digits=8)
        acc = accuracy_score(responses, y_predict)
        f1 = f1_score(responses, y_predict, labels=labels, average='weighted')
        return {
            "acc": acc,
            "f1": f1,
            "confusion_matrix": my_confusion_matrix,
            "classification_report": my_report
        }
    else:
        responses = [float(response) for response in responses]
        responses = asarray(responses)
        evs = explained_variance_score(responses, y_predict)
        mae = mean_absolute_error(responses, y_predict)
        mse = mean_squared_error(responses, y_predict)
        r2 = r2_score(responses, y_predict)
        return {
            "Explained variance": evs,
            "Mean absolute error": mae,
            "Mean squared error": mse,
            "R2": r2
        }


def get_features_response(feature_path, response_path):
    """
    Get the feature array and response list from files.

    Parameters
    ----------
    feature_path: str
        The location of the features file.
    response_path: str
        The location of the response file.

    Returns
    -------
    array, list<str>
        A numpy array of features and a list of responses.

    """
    features = []
    responses = []
    if feature_path:
        with open(feature_path) as feature_file:
            skip_first = HEADER_EXISTS
            for line in feature_file:
                if skip_first:
                    skip_first = False
                    continue
                observation = list(map(float, line.strip().split()))
                features.append(observation)
    if response_path:
        with open(response_path) as response_file:
            responses = [line.strip() for line in response_file]
    return asarray(features), responses


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
    with open(os.path.join(model_path, NAME_PATH), 'r') as name_file:
        name = name_file.readline().strip().lower()
    return model, encoder, name


def __get_feature_imp(model, name=None):
    if name:
        if ESTIMATOR_CHOICES[name][1] == 0:
            return model.named_steps[name].feature_importances_
        else:
            return model.named_steps[name].coef_
    else:
        for estimator in ESTIMATOR_CHOICES:
            if estimator in model.named_steps:
                if ESTIMATOR_CHOICES[estimator][1] == 0:
                    return model.named_steps[estimator].feature_importances_
                else:
                    return model.named_steps[estimator].coef_


def save_model(model, encoder, name, model_path):
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
    name = name.lower()
    pickle.dump(model, open(os.path.join(model_path, MODEL_NAME), 'wb'))
    pickle.dump(encoder, open(os.path.join(model_path, ENCODER_NAME), 'wb'))
    print(name, file=open(os.path.join(model_path, NAME_PATH), 'w'))
    print(str(__get_feature_imp(model, name)),
          file=open(os.path.join(model_path, FEATURE_COEFF), 'w'))


class ArgClass:
    """An argument class for commands."""
    def __init__(self, *args, **kwargs):
        """Store arguments."""
        self.args = args
        self.kwargs = kwargs


def main():
    """Parse arguments."""
    tick = datetime.datetime.now()
    parser = argparse.ArgumentParser(
        description=("Train, predict, and "
                     "evaluate a scikit learn "
                     "estimators using "
                     "Uses bayesian optimization "
                     "for hyperparameter tuning. "),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    estimator = ArgClass(
        "-e",
        "--estimator",
        type=str,
        choices=[ESTIMATOR_CHOICES[c][2] for c in ESTIMATOR_CHOICES],
        help=("The estimator to use."),
        default='RandomForestClassifier')
    iterations = ArgClass("-i",
                          "--iterations",
                          type=int,
                          help="Iterations of Bayesian optimization to do.",
                          default=25)
    folds = ArgClass("-f",
                     "--folds",
                     type=int,
                     help="The number of folds of cross validation to do.",
                     default=3)
    processes = ArgClass("-p",
                         "--processes",
                         type=int,
                         help="The number of processes to do training on.",
                         default=3)
    verbose = ArgClass("-v",
                       "--verbose",
                       type=int,
                       help="Controls the verbosity of the estimator.",
                       default=10)
    subparsers = parser.add_subparsers(help="sub-commands", dest="mode")
    parser_train = subparsers.add_parser(
        "train",
        help="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.add_argument("model_path",
                              type=str,
                              help=("The location where the "
                                    "model is stored. "))
    parser_train.add_argument("feature_path",
                              type=str,
                              help=("The location of the features data."))
    parser_train.add_argument("response_path",
                              type=str,
                              help=("The location of the response data."))
    parser_train.add_argument(*estimator.args, **estimator.kwargs)
    parser_train.add_argument(*iterations.args, **iterations.kwargs)
    parser_train.add_argument(*folds.args, **folds.kwargs)
    parser_train.add_argument(*processes.args, **processes.kwargs)
    parser_train.add_argument(*verbose.args, **verbose.kwargs)
    parser_predict = subparsers.add_parser(
        "predict",
        help=("Make predictions "
              "given a saved model. "
              "The predictions are "
              "written to stdout."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_predict.add_argument("model_path",
                                type=str,
                                help=("The location where the "
                                      "model is stored. "))
    parser_predict.add_argument("feature_path",
                                type=str,
                                help=("The location of the features data."))
    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help=("Make predictions and "
              "evaluate their predictive "
              "performance.  "
              "The evaluation is "
              "written to stdout. "),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_evaluate.add_argument("model_path",
                                 type=str,
                                 help=("The location where the "
                                       "model is stored. "))
    parser_evaluate.add_argument("feature_path",
                                 type=str,
                                 help=("The location of the features data."))
    parser_evaluate.add_argument("response_path",
                                 type=str,
                                 help=("The location of the response data."))
    args = parser.parse_args()
    print(args, file=sys.stderr)
    mode = args.mode
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if mode == "train":
        features, response = get_features_response(args.feature_path,
                                                   args.response_path)
        model, encoder = train(features,
                               response,
                               args.model_path,
                               estimator=args.estimator,
                               iterations=args.iterations,
                               folds=args.folds,
                               processes=args.processes,
                               verbose=args.verbose)
        save_model(model, encoder, args.estimator, args.model_path)
    elif mode == "predict":
        features, _ = get_features_response(args.feature_path, None)
        model, encoder, name = load_model(args.model_path)
        is_regressor = ESTIMATOR_CHOICES[name.lower()][3]
        responses, proba, order = predict(model, name, features, encoder)
        if not is_regressor and proba is not None:
            print("{}\t{}".format("predicted_class", "\t".join(order)))
            for label, p in zip(responses, proba):
                print("{}\t{}".format(label,
                                      str(p).replace("[", "").replace("]",
                                                                      "")))
        else:
            print("Response")
            for item in responses:
                print(str(item))
    elif mode == "evaluate":
        features, responses = get_features_response(args.feature_path,
                                                    args.response_path)
        model, encoder, name = load_model(args.model_path)
        evaluation_reports = evaluate(model, name, features, responses,
                                      encoder)
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
