import numpy as np
import pandas as pd
import datetime
import nltk
import shap
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SVMSMOTE
from anchor import anchor_tabular


def get_stop_words():
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.remove('not')
    return list(stopwords)


def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def preprocess(text):
    stemmer = nltk.stem.PorterStemmer()
    stopwords = get_stop_words()
    word_bag = list()
    for word in nltk.word_tokenize(text):
        if word in stopwords:
            word_bag.append(word)
        else:
            word_bag.append(stemmer.stem(word))

    return " ".join(word_bag)


def preprocess_lem(text):
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    stopwords = get_stop_words()
    word_bag = list()
    for word in nltk.word_tokenize(text):
        if word in stopwords:
            word_bag.append(word)
        else:
            word_bag.append(lem.lemmatize(word, get_wordnet_pos(word)))

    return " ".join(word_bag)


downsample_proportion = 1.0
upsample_multiplier = 3
#anchor_threshold = 0.8
voting_method = 'soft'
oversamplers = {'svmsmote': SVMSMOTE()}
nlp = spacy.load('en_core_web_sm')

xlsx_path = 'launchpad_bugs/ubuntu_launchpad_bugs.xlsx'
xlsx_reports = pd.read_excel(xlsx_path, dtype={'Bug ID': str, 'Title': str, 'Date Created': str,
                                               'Date Confirmed': str, 'Date Closed': str,
                                               'Web Link': str, 'Patch Link': str, 'Importance': str,
                                               'Security Bug': str, 'Description': str, 'Activity': str,
                                               'Trigger': str, 'Impact': str, 'Target': str,
                                               'Defect Type': str, 'Qualifier': str, 'CRASH': str})


null_values = pd.isnull(xlsx_reports['Title'])
null_values2 = pd.isnull(xlsx_reports['Description'])
null_values = null_values.index[null_values.values].tolist()
null_values2 = null_values2.index[null_values2.values].tolist()
null_values = list(set(null_values + null_values2))

clean_reports = xlsx_reports.drop(index=null_values)
security_tags = ['bug', 'Yes', 'Positive']
other_tags = ['non-bug', 'No', 'Negative']
positive_class = 'Positive'
negative_class = 'Negative'
classes = [positive_class, negative_class]
le = LabelEncoder()
le.fit(classes)
MARKER = '\n#--------------------------#\n'
feature_amount = 125
anchor_threshold = 0.95

clean_reports['Outputs'] = np.where(clean_reports['Security Bug'].isin(security_tags), positive_class, negative_class)
clean_reports['Title'] = clean_reports['Title'].apply(lambda x: re.sub('\d', '', x))
clean_reports['Description'] = clean_reports['Description'].apply(lambda x: re.sub('\d', '', x))
stop_words = get_stop_words()

for input_feature in ['Title']:
    for oversampler_name in oversamplers:
        train_X, test_X, train_Y, test_Y = train_test_split(clean_reports[input_feature], clean_reports['Outputs'],
                                                            test_size=0.2)

        results = pd.DataFrame(columns=['Classifier', 'Input', 'Test Phase', 'Fold', 'Accuracy', 'Precision', 'F-score',
                                        'Recall', 'True Positives', 'False Positives', 'False Negatives',
                                        'True Negatives'])

        kf = KFold(n_splits=10)
        fold_count = 1

        feature_selection_estimator = LinearSVC()

        preprocessor = Pipeline([('tfidf', TfidfVectorizer(preprocessor=preprocess, stop_words=stop_words)),
                                 ('featsel', SelectFromModel(estimator=LinearSVC(), max_features=feature_amount,
                                                             threshold='median'))])
        preprocessor.fit(train_X, train_Y)

        classifier = LogisticRegression(max_iter=500)

        for train, test in kf.split(train_X):
            current_inputs = np.array(train_X)[train]
            current_inputs = preprocessor.transform(current_inputs).toarray()
            current_outputs = np.array(train_Y)[train]

            test_input_text = np.array(train_X)[test]
            test_input = preprocessor.transform(test_input_text).toarray()
            test_output = np.array(train_Y)[test]
            positive_indexes = np.where(test_output == positive_class)

            oversampler = oversamplers[oversampler_name]
            current_inputs, current_outputs = oversampler.fit_resample(current_inputs, current_outputs)

            print('Training Logistic Regression model...')
            classifier.fit(current_inputs, current_outputs)
            predicted = classifier.predict(test_input)

            print('Taking model coefficients...')
            lr_coefficients = pd.DataFrame.from_dict({'Word': preprocessor.get_feature_names_out(),
                                                      'Coefficient': classifier.coef_[0]})
            lr_coefficients.to_csv('explanation_logs/coef_fold{}_{}.csv'.format(fold_count, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

            print('Taking SHAP explanations...')
            shap_explainer = shap.Explainer(classifier, current_inputs, feature_names=preprocessor.get_feature_names_out())
            shap_values = shap_explainer(test_input)
            shap_explanations = pd.DataFrame(columns=shap_values.feature_names, data=shap_values.data)
            mul_condition = np.where(predicted == positive_class, 1, -1)
            for feature in shap_values.feature_names:
                shap_explanations[feature] = shap_explanations[feature] * mul_condition
            shap_explanations.to_csv('explanation_logs/shap_fold{}_{}.csv'.format(fold_count, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

            def predict_lr(texts):
                return classifier.predict(preprocessor.transform(texts))

            print('Taking Anchor explanations...')
            anchor_explainer = anchor_tabular.AnchorTabularExplainer(classes, preprocessor.get_feature_names_out(),
                                                                     current_inputs)

            anchor_explanations = pd.DataFrame(columns=['Prediction', 'Anchors', 'Precision'])

            for explaination_input in test_input:

                prediction = classifier.predict([explaination_input])
                exp = anchor_explainer.explain_instance(explaination_input, classifier.predict,
                                                        threshold=anchor_threshold)
                anchor_explanations = pd.concat([anchor_explanations, pd.DataFrame.from_dict(
                    {'Prediction': [prediction[0]],
                     'Anchors': [(' AND '.join(exp.names()))],
                     'Precision': exp.precision()})])
                print(MARKER)
                print('Prediction: %s' % prediction[0])
                print('Anchor: %s' % (' AND '.join(exp.names())))
                print('Precision: %.2f' % exp.precision())
                print(MARKER)

            anchor_explanations.to_csv('explanation_logs/anchor_fold{}_{}.csv'.format(fold_count, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

            print('\nFold:{} Logistic Regression accuracy:{}'.format(fold_count, np.mean(predicted == test_output)))
            print(metrics.classification_report(test_output, predicted, labels=classes))
            lr_metrics = metrics.classification_report(test_output, predicted, labels=classes, output_dict=True)
            lr_confusion_matrix_data = metrics.confusion_matrix(test_output, predicted, labels=classes)
            results = pd.concat([results, pd.DataFrame.from_dict({'Classifier': ['Logistic Regression'],
                                                                  'Input': [input_feature],
                                                                  'Test Phase': ['Validation'],
                                                                  'Fold': [fold_count],
                                                                  'Accuracy': [lr_metrics['accuracy']],
                                                                  'Precision': [lr_metrics[positive_class]['precision']],
                                                                  'F-score': [lr_metrics[positive_class]['f1-score']],
                                                                  'Recall': [lr_metrics[positive_class]['recall']],
                                                                  'True Positives': [lr_confusion_matrix_data[0][0]],
                                                                  'False Positives': [lr_confusion_matrix_data[0][1]],
                                                                  'False Negatives': [lr_confusion_matrix_data[1][0]],
                                                                  'True Negatives': [lr_confusion_matrix_data[1][1]]})])
            fold_count += 1

        predicted = classifier.predict(preprocessor.transform(test_X))
        print('Test Logistic Regression accuracy:{}'.format(np.mean(predicted == test_Y)))
        print(metrics.classification_report(test_Y, predicted, labels=classes))
        lr_metrics = metrics.classification_report(test_Y, predicted, labels=classes, output_dict=True)
        lr_confusion_matrix_data = metrics.confusion_matrix(test_Y, predicted, labels=classes)
        results = pd.concat([results, pd.DataFrame.from_dict({'Classifier': ['Logistic Regression'],
                                                              'Input': [input_feature],
                                                              'Test Phase': ['Test'],
                                                              'Fold': [0],
                                                              'Accuracy': [lr_metrics['accuracy']],
                                                              'Precision': [lr_metrics[positive_class]['precision']],
                                                              'F-score': [lr_metrics[positive_class]['f1-score']],
                                                              'Recall': [lr_metrics[positive_class]['recall']],
                                                              'True Positives': [lr_confusion_matrix_data[0][0]],
                                                              'False Positives': [lr_confusion_matrix_data[0][1]],
                                                              'False Negatives': [lr_confusion_matrix_data[1][0]],
                                                              'True Negatives': [lr_confusion_matrix_data[1][1]]})])

        results.reset_index()
        results.to_csv('results_ubuntu_oversampling/{}/results_{}_dim{}_{}.csv'.format(oversampler_name,
                                                                                       input_feature,
                                                                                       feature_amount,
                                                                                       datetime.datetime.now().
                                                                                       strftime("%Y%m%d-%H%M%S")))
