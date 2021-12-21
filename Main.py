import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler
import random
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import pickle


def trivial_system(X_train, y_train, X_test, y_test):
    dataset = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    c1 = []
    c2 = []
    c3 = []
    for i in range(len(dataset)):
        if dataset[i, -1] == 1:
            c1.append(i)
        else:
            if dataset[i, -1] == 2:
                c2.append(i)
            else:
                if dataset[i, -1] == 3:
                    c3.append(i)
    N = len(X_train)
    p1 = len(c1)/N
    p2 = len(c2)/N
    p3 = len(c3)/N
    label = [1, 2, 3]
    idx_test = list(range(len(X_test)))
    y_pred = np.random.choice(label, len(X_test), replace=True, p=[p1, p2, p3])
    c = 0
    for j in range(len(y_test)):
        if y_pred[j] == y_test[j]:
            c += 1
    accuracy = c / len(y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    CF_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    print('Confusion Matrix of the trivial system:\n', CF_matrix)
    plt.show()
    return accuracy, f1


def undersample(features, label, size):
    # get the concat dataset
    dataset = np.concatenate([features, label.reshape(-1, 1)], axis=1)
    c1 = []
    c2 = []
    c3 = []
    for i in range(len(dataset)):
        if dataset[i, -1] == 1:
            c1.append(i)
        else:
            if dataset[i, -1] == 2:
                c2.append(i)
            else:
                if dataset[i, -1] == 3:
                    c3.append(i)

    # get the new undersample dataset
    reduced_1 = []
    reduced_2 = []
    reduced_3 = []
    a1 = np.random.choice(c1, size=size)
    a2 = np.random.choice(c2, size=size)
    a3 = np.random.choice(c3, size=size)
    for i in range(len(a1)):
        reduced_1.append(dataset[i])
    for i in range(len(a2)):
        reduced_2.append(dataset[i])
    for i in range(len(a3)):
        reduced_3.append(dataset[i])

    reduced = np.concatenate([reduced_1, reduced_2, reduced_3], axis=0)
    reduced_feature = reduced[:, :-1]
    reduced_label = reduced[:, -1]
    return reduced_feature, reduced_label


def oversample(features, label):
    # get the concat dataset
    dataset = np.concatenate([features, label.reshape(-1, 1)], axis=1)
    c1 = []
    c2 = []
    c3 = []
    for i in range(len(dataset)):
        if dataset[i, -1] == 1:
            c1.append(i)
        else:
            if dataset[i, -1] == 2:
                c2.append(i)
            else:
                if dataset[i, -1] == 3:
                    c3.append(i)

    # get the new oversample dataset
    size = np.max([len(c1), len(c2), len(c3)])
    if size == len(c1):
        c1_new = c1
        c2_new = resample(c2, replace=True, n_samples=size, random_state=None)
        c3_new = resample(c3, replace=True, n_samples=size, random_state=None)
    else:
        if size == len(c2):
            c1_new = resample(c1, replace=True, n_samples=size, random_state=None)
            c2_new = c2
            c3_new = resample(c3, replace=True, n_samples=size, random_state=None)
        else:
            if size == len(c3):
                c1_new = resample(c1, replace=True, n_samples=size, random_state=None)
                c2_new = resample(c2, replace=True, n_samples=size, random_state=None)
                c3_new = c3

    over_1 = []
    over_2 = []
    over_3 = []
    for i in range(size):
        over_1.append(dataset[c1_new[i], :])
    for i in range(size):
        over_2.append(dataset[c2_new[i], :])
    for i in range(size):
        over_3.append(dataset[c3_new[i], :])

    dataset_oversampled = np.concatenate([over_1, over_2, over_3], axis=0)
    features_oversampled = dataset_oversampled[:, :-1]
    labels_oversampled = dataset_oversampled[:, -1]

    return features_oversampled, labels_oversampled


def CrossValidation(clf, skf, features_train, label_train):
    n = skf.get_n_splits(features_train, label_train)
    temp = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(features_train, label_train, test_size=0.2, train_size=0.8,
                                                            shuffle=True)
        # for train_index, test_index in skf.split(features_train, label_train):
        #     X_train, X_test = features_train[train_index], features_train[test_index]
        #     y_train, y_test = label_train[train_index], label_train[test_index]

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        temp.append(score)
    all_accuracy = temp
    mean_accuracy = np.mean(temp)
    std_accracy = np.std(temp)

    return all_accuracy, mean_accuracy, std_accracy


def Standardize_Noramalize(model, data):
    standardized_data = model.fit_transform(data)
    return standardized_data


def perceptron_normalized(X_train, y_train):
    regr_model = Pipeline([

                            # ('trans', PolynomialFeatures(degree=2)),
                            # ('pca', PCA(n_components=50)),
                            ('scaler', StandardScaler()),

                            # ('mms', MinMaxScaler()),
                            # ('mas', MaxAbsScaler()),
                           ('clf', Perceptron())])
    regr_model.fit(X_train, y_train)
    return regr_model


def LDA_normalized(X_train, y_train):
    lda_model = Pipeline([
        ('trans', PolynomialFeatures(degree=2)),
        # ('pca', PCA(n_components=10)),
        # ('scaler', StandardScaler()),
        #  ('mms', MinMaxScaler()),
        # ('mas', MaxAbsScaler()),
        # ('nor', Normalizer()),
        ('clf', LinearDiscriminantAnalysis(priors=[622/2961, 1339/2961, 1000/2961]))])
    lda_model.fit(X_train, y_train)
    return lda_model


def KNN_normalized(X_train, y_train):
    knn_model = Pipeline([
        # ('trans', PolynomialFeatures(degree=2)),
        # ('pca', PCA(n_components=10, svd_solver='a')),
        # ('scaler', StandardScaler()),
        #  ('mms', MinMaxScaler()),
        # ('mas', MaxAbsScaler()),
        ('clf', KNeighborsClassifier())])
    knn_model.fit(X_train, y_train)
    return knn_model


def SVM_normalized(X_train, y_train):
    svm_model = Pipeline([
        ('trans', PolynomialFeatures(degree=2)),
        # ('pca', PCA(n_components=10, svd_solver='a')),
        # ('scaler', StandardScaler()),
         ('mms', MinMaxScaler()),
        # ('mas', MaxAbsScaler()),
        ('clf', KNeighborsClassifier())])
    svm_model.fit(X_train, y_train)
    return svm_model


def show_result(modelname, model, test, testlabel, CFmatrix):
    # print(train.shape)
    # print(trainlabel.shape)
    # print(test)
    # print(test.shape)
    print('===' * 20)
    print('\n> ', modelname)
    predict = model.predict(test)
    score = model.score(test, testlabel)
    print('\n>  Test accuracy is: ', score)
    if CFmatrix is True:
        CF_matrix = confusion_matrix(testlabel, predict)
        print(CF_matrix)
        plot_confusion_matrix(model, test, testlabel, cmap=plt.cm.Blues, normalize='true')
        plt.title('confusion matrix for {0}'.format(modelname))
        plt.xlabel('predict label')
        plt.ylabel('ground truth')
        plt.show()
    print(classification_report(testlabel, predict, zero_division=0, digits=5))
    return predict


if __name__ == "__main__":

    plt.ion()

    '''load data'''
    # load data
    wine_train_features = pd.read_csv('WINE_Training_data.csv').values
    wine_train_label = pd.read_csv('WINE_Training_label.csv').values.ravel()
    wine_test_features = pd.read_csv('WINE_Test_data.csv').values
    wine_test_label = pd.read_csv('WINE_Test_label.csv').values.ravel()

    '''Pre-processing'''
    # standardized data
    sca = StandardScaler()
    wine_train_features_st = sca.fit_transform(wine_train_features)
    wine_test_features_st = sca.transform(wine_test_features)

    # normalized data
    mms = MinMaxScaler()
    wine_train_features_mms = mms.fit_transform(wine_train_features)
    wine_test_features_mms = mms.transform(wine_test_features)

    # normalized data
    mas = MaxAbsScaler()
    wine_train_features_mas = mas.fit_transform(wine_train_features)
    wine_test_features_mas = mas.transform(wine_test_features)

    # data balance
    sm = SMOTE()  # sampling_strategy='minority', k_neighbors=4
    train_resample, train_label_resample = sm.fit_resample(wine_train_features, wine_train_label)

    sca = StandardScaler()
    train_resample_st = sca.fit_transform(train_resample)
    test_resample_st = sca.transform(wine_test_features)

    '''Feature Engineering'''
    ply = PolynomialFeatures(degree=2)
    train_expand = ply.fit_transform(wine_train_features)
    test_expand = ply.transform(wine_test_features)

    '''Feature dimensionality adjustment'''
    # feature selection by KBEST
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest
    MI = mutual_info_classif(wine_train_features, wine_train_label)
    MI2 = mutual_info_classif(train_expand, wine_train_label)
    print('Mutual Information:', MI)
    print('Mutual Information for expansion feature:', MI2)
    plt.figure()
    X = np.arange(11) + 1
    plt.bar(X,MI)
    plt.title('Mutual Information based on original feature space')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')

    plt.figure()
    X2 = np.arange(78) + 1
    plt.bar(X2,MI2)
    plt.title('Mutual Information based on expansion feature space')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')

    from sklearn.feature_selection import chi2
    chi2val, pval = chi2(wine_train_features_mms, wine_train_label)
    print('chi2, pvalue:',chi2val, pval)

    skb = SelectKBest(score_func=chi2, k=3)  # 8 is good
    train_selected = skb.fit_transform(wine_train_features_mms, wine_train_label)
    test_selected = skb.transform(wine_test_features_mms)

    # check feature importance by RF
    rfcf1 = RandomForestClassifier()
    rfcf1.fit(wine_train_features, wine_train_label)
    fi = rfcf1.feature_importances_
    print('Feature Importance based on original feature space', fi)
    Xi = np.arange(11) + 1
    plt.figure()
    plt.bar(Xi, fi)
    plt.title('Feature Importance based on original feature space')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    rfcf2 = RandomForestClassifier()
    rfcf2.fit(train_expand, wine_train_label)
    fi2 = rfcf2.feature_importances_
    print('Feature Importance based on expansion feature space', fi2)
    Xi2 = np.arange(78) + 1
    plt.figure()
    plt.bar(Xi2, fi2)
    plt.title('Feature Importance based on expansion feature space')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    '''PCA'''
    # # Crossvalidation for PCA best parameters
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # per = Perceptron()
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(1, 10):
    #     pca_cv = PCA(n_components=i)
    #     train_pca = pca_cv.fit_transform(wine_train_features)
    #     # sca = StandardScaler()
    #     # train_pca = sca.fit_transform(train_pca)
    #     result, mean, std = CrossValidation(per, skf, train_pca, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    #
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    #
    # idx = np.argmax(mean_score)
    # print(mean_score)
    # print('best number of PCA component is: ', idx+1)
    #
    #
    # pca = PCA(n_components=8)
    # wine_train_features_pca = pca.fit_transform(wine_train_features)
    # wine_test_features_pca = pca.transform(wine_test_features)
    # clfpca = perceptron_normalized(wine_train_features_pca, wine_train_label)  # 最好的是mas
    # show_result('Perceptron_normalized testset after PCA', clfpca, wine_test_features_pca, wine_test_label)
    # print('Perceptron PCA train accuracy is: ', clfpca.score(wine_train_features_pca, wine_train_label))


    # '''oversample / undersample'''
    # wine_train_features_os, wine_train_label_os = oversample(wine_train_features, wine_train_label)
    # wine_train_features_us, wine_train_label_us = undersample(wine_train_features, wine_train_label, 600)
    #
    # # std oversample
    # sca = StandardScaler()
    # wine_train_features_os_st = sca.fit_transform(wine_train_features_os)
    # wine_test_features_os_st = sca.transform(wine_test_features)
    #
    # # mas oversample
    # mas = MaxAbsScaler()
    # wine_train_features_os_mas = mas.fit_transform(wine_train_features_os)
    # wine_test_features_os_mas = mas.transform(wine_test_features)
    #
    # # mms oversample
    # mms = MinMaxScaler()
    # wine_train_features_os_mms = mms.fit_transform(wine_train_features_os)
    # wine_test_features_os_mms = mms.transform(wine_test_features)

    '''Trivial System'''
    print('==='*20)
    trival_acc, f1 = trivial_system(wine_train_features, wine_train_label, wine_test_features, wine_test_label)

    print('Test accuracy of the trivial system based on probability is: ', trival_acc)
    print('f1 score of the trivial system based on probability is: ', f1)
    # trival_train_acc, _ = trivial_system(wine_train_features, wine_train_label, wine_train_features, wine_train_label)
    # print('Train accuracy of the trivial system based on probability is: ', trival_train_acc)

    '''Baseline: Default Perceptron with Standardization dataset'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # change the Perceptron_normalized function to use different pre-processing method
    # per_st = Perceptron()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     m = StandardScaler()  # change this to compare the result of Standardization/Normalization for LDA
    #     # m = MaxAbsScaler()
    #     # m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)  # wine_train_features
    #     result, mean, std = CrossValidation(per_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of Perceptron is:', np.mean(mean_score))
    # print('validation std', np.mean(std_score))

    # ======================================================================== #
    # part 2: Baseline model, use default Perceptron and standardized data

    clf1 = perceptron_normalized(wine_train_features, wine_train_label)  # change the setting in function Perceptron_normalized
    show_result('Baseline: Perceptron with standardized dataset', clf1, wine_test_features, wine_test_label, CFmatrix=True)
    print('Perceptron with standardized dataset train accuracy is: ', clf1.score(wine_train_features, wine_train_label))

    '''feature expansion degree CV by LDA'''
    # LDA CrossValidation for best expansion degree

    # lda4 = LinearDiscriminantAnalysis()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # degr = [1,2,3,4,5]
    # for i in range(len(degr)):
    #     trans = PolynomialFeatures(degree=degr[i])
    #     data = trans.fit_transform(wine_train_features, wine_train_label)
    #     result, mean, std = CrossValidation(lda4, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    #
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    #
    # idx = np.argmax(mean_score)
    # print('==='*20)
    # print('result of different degree', mean_score)
    # print('best expansion degree is: ', degr[idx])
    # plt.figure()
    # plt.title('Validation Accuracy - Feature Expansion Degree')
    # plt.plot(degr, mean_score)
    # plt.xticks([1,2,3,4,5])
    # plt.xlabel('Feature Expansion Degree')
    # plt.ylabel('Validation Accuracy')
    # # conclusion: best is degree=2

    '''Model 1: LDA'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # # change the LDA_normalized function to use different pre-processing method
    # lda_st = LinearDiscriminantAnalysis()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     # m = StandardScaler()  # change this to compare the result of Standardization/Normalization for LDA
    #     # m = MaxAbsScaler()
    #     m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)
    #     result, mean, std = CrossValidation(lda_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of LDA is:', np.mean(mean_score))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion

    # expanded performer better
    # lda2 = LinearDiscriminantAnalysis()
    # train_expand = ply.fit_transform(wine_train_features)
    # test_expand = ply.transform(wine_test_features)
    #
    # lda2.fit(train_expand, wine_train_label)
    # show_result('LDA feature expansion', lda2, test_expand, wine_test_label)
    # print('LDA train accuracy is: ', lda2.score(train_expand, wine_train_label))

    # ======================================================================== #
    # <TEST> part 3: LDA best parameter CV

    ply = PolynomialFeatures(degree=2)
    train_expand_resample = ply.fit_transform(train_resample)
    test_expand = ply.transform(wine_test_features)

    param_grid_lda = [{'solver': ["svd", "lsqr"],
                       }]
    grid_lda_clf = LinearDiscriminantAnalysis()
    grid_search_lda = GridSearchCV(grid_lda_clf, param_grid_lda, cv=5, verbose=1)
    grid_search_lda.fit(train_expand_resample, train_label_resample)
    clf_lda = grid_search_lda.best_estimator_
    clf_lda.fit(train_expand_resample, train_label_resample)

    show_result('Model 1: LDA best model', clf_lda, test_expand, wine_test_label, CFmatrix=False)
    print('LDA best model train accuracy is: ', clf_lda.score(train_expand_resample, train_label_resample))
    print('LDA best parameter is: ', grid_search_lda.best_params_)
    print('LDA best model val accuracy', grid_search_lda.best_score_)
    print('cv_result:', grid_search_lda.cv_results_)

    # ======================================================================== #
    # part 4: final best model on LDA test
    # use the oversampled data + feature expansion (degree=2) to achieve the best

    lda_final = LinearDiscriminantAnalysis()
    lda_final.fit(train_expand_resample, train_label_resample)
    show_result('Model 1: LDA final testset', lda_final, test_expand, wine_test_label, CFmatrix=True)
    print('LDA train accuracy is: ', lda_final.score(train_expand_resample, train_label_resample))

    '''Model 2: KNN'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # knn_st = KNeighborsClassifier()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     m = StandardScaler()  # change this to compare the result of Standardization/Normalization for KNN
    #     # m = MaxAbsScaler()
    #     # m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)  # wine_train_features
    #     result, mean, std = CrossValidation(knn_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of KNN is:', np.mean(mean_score))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion
    # to see the difference of feature expansion

    # expanded performs better
    ply = PolynomialFeatures(degree=2)
    train_expand = ply.fit_transform(wine_train_features)
    test_expand = ply.transform(wine_test_features)

    train_expand_st = sca.fit_transform(train_expand)
    test_expand_st = sca.transform(test_expand)

    train_expand_mms = mms.fit_transform(train_expand)
    test_expand_mms = mms.transform(test_expand)

    train_expand_mas = mas.fit_transform(train_expand)
    test_expand_mas = mas.transform(test_expand)

    train_expand_resample = ply.fit_transform(train_resample)
    test_expand = ply.transform(wine_test_features)

    # knn2 = KNeighborsClassifier()
    # knn2.fit(train_expand_resample, train_label_resample)
    # show_result('normal Knn test', knn2, test_expand, wine_test_label, CFmatrix=False)
    # print('Knn train accuracy is: ', knn2.score(train_expand_resample, train_label_resample))

    # ======================================================================== #
    #  <TEST> part 3: KNN best parameter CV

    param_grid_knn = [{
                       'n_neighbors': [1,2,5,8,10,15,20,50,60,65,70,75,80,90]}]

    grid_knn_clf = KNeighborsClassifier(n_jobs=-1)
    grid_search_knn = GridSearchCV(grid_knn_clf, param_grid_knn, cv=5, verbose=1)
    grid_search_knn.fit(train_expand_st, wine_train_label)
    print(grid_search_knn.cv_results_)
    clf_knn = grid_search_knn.best_estimator_
    clf_knn.fit(train_expand_st, wine_train_label)

    show_result('KNN best model', clf_knn, test_expand_st, wine_test_label, CFmatrix=False)
    print(grid_search_knn.best_params_)
    print('KNN val accuracy', grid_search_knn.best_score_)
    # KNN best is:{'n_neighbors': 65, 'weights': 'distance'}

    # ======================================================================== #
    # part 4: final best model on KNN test
    # imbalance + expanded feature space (degree=2) + standardization / K=60

    knn2 = KNeighborsClassifier(n_neighbors=60, weights='distance')
    knn2.fit(train_expand_st, wine_train_label)
    show_result('Model 2: KNN final testset', knn2, test_expand_st, wine_test_label, CFmatrix=True)
    print('Knn train accuracy is: ', knn2.score(train_expand_st, wine_train_label))

    '''Model 3: OVR SVC'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # change the SVM_normalized function to use different pre-processing method
    # svm_st = SVC()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     m = StandardScaler()  # change this to compare the result of Standardization/Normalization for SVM
    #     # m = MaxAbsScaler()
    #     # m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)  # wine_train_features
    #     result, mean, std = CrossValidation(svm_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of SVM is:', np.mean(mean_score))

    # compare imbalance & balance
    # svm = SVC(decision_function_shape='ovr')
    # clf_svm = svm.fit(wine_train_features, wine_train_label)
    # show_result('Model 3: OVR SVC test', svm, wine_test_features, wine_test_label, CFmatrix=True)
    # print('OVR SVM training accuracy', svm.score(wine_train_features, wine_train_label))
    #
    # svm = SVC(decision_function_shape='ovr')
    # svm.fit(train_resample, train_label_resample)
    # show_result('Model 3: OVR SVC test resample', svm, wine_test_features, wine_test_label, CFmatrix=True)
    # print('OVR SVM training accuracy', svm.score(train_resample, train_label_resample))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion
    # to see the difference of feature expansion

    # expanded performs better
    # svm2 = SVC()
    # svm2.fit(train_expand, wine_train_label)
    # show_result(' SVM expansion test', svm2, test_expand, wine_test_label, CFmatrix=False)
    # print('SVM train accuracy is: ', svm2.score(train_expand, wine_train_label))
    #
    # svm2 = SVC()
    # svm2.fit(wine_train_features, wine_train_label)
    # show_result(' SVM unexpansion test', svm2, wine_test_features, wine_test_label, CFmatrix=False)
    # print('SVM train accuracy is: ', svm2.score(wine_train_features, wine_train_label))

    # ======================================================================== #
    #  <TEST> part 3: SVM best parameter CV
    ply = PolynomialFeatures(degree=2)
    train_expand_resample = ply.fit_transform(train_resample)
    test_expand = ply.transform(wine_test_features)
    sca = StandardScaler()
    train_expand_resample_st = sca.fit_transform(train_expand_resample)
    test_expand_resample_st = sca.transform(test_expand)

    s1 = pickle.dumps(sca)
    with open('model_sca.pkl', 'wb+') as f:
        f.write(s1)
    test_expand_resample_st = sca.transform(test_expand)


    # best is: {'C': 268.26957952797244, 'decision_function_shape': 'ovr', 'gamma': 0.0071968567300115215}
    param_grid_svm = [
        {'gamma': np.logspace(-2, 2, num=15).tolist(),
         'decision_function_shape': ["ovr"],
         'C': np.logspace(-2, 2, num=15).tolist(),
         }]
    grid_svm = SVC()
    grid_search_svm = GridSearchCV(grid_svm, param_grid_svm, cv=5, verbose=1)
    grid_search_svm.fit(wine_train_features_st, wine_train_label)
    print(grid_search_svm.cv_results_)
    svm = grid_search_svm.best_estimator_
    svm.fit(train_expand_resample_st, train_label_resample)
    show_result('Model 3: OVR SVM', svm, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('Model 3: OVR SVM training accuracy', svm.score(train_expand_resample_st, train_label_resample))
    print(grid_search_svm.best_params_)
    print('OVR SVM val accuracy', grid_search_svm.best_score_)

    # ======================================================================== #
    # part 4: final best model on SVM test
    # oversampel + feature expansion (degree=2) + Standardization/ C = 150, gamma = 0.001, rbf, ovr

    svm_final = SVC(C=150, gamma=0.001, kernel='rbf', decision_function_shape='ovr')
    svm_final.fit(train_expand_resample_st, train_label_resample)
    show_result('Model 3: SVM final testset', svm_final, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('SVM train accuracy is: ', svm_final.score(train_expand_resample_st, train_label_resample))

    '''Model 4: SGD Classifier'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # change the SVM_normalized function to use different pre-processing method
    # sgd_st = SGDClassifier()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     # m = StandardScaler()  # change this to compare the result of Standardization/Normalization for SGD
    #     # m = MaxAbsScaler()
    #     m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)  # wine_train_features
    #     result, mean, std = CrossValidation(sgd_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of SGD is:', np.mean(mean_score))

    # compare imbalance & balance
    # quite unstable, may use balance
    # sgd = SGDClassifier()
    # sgd.fit(wine_train_features, wine_train_label)
    # show_result('Model 4: sgd test', sgd, wine_test_features, wine_test_label, CFmatrix=False)
    # print('SGD training accuracy', sgd.score(wine_train_features, wine_train_label))
    #
    # sgd = SGDClassifier()
    # sgd.fit(train_resample, train_label_resample)
    # show_result('Model 4:sgd resample test resample', sgd, wine_test_features, wine_test_label, CFmatrix=False)
    # print('SGD training accuracy', sgd.score(train_resample, train_label_resample))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion
    # to see the difference of feature expansion

    # quite unstable, maybe use expand
    # sgd2 = SGDClassifier()
    # sgd2.fit(train_expand, wine_train_label)
    # show_result(' SGD expansion test', sgd2, test_expand, wine_test_label, CFmatrix=False)
    # print('SGD train accuracy is: ', sgd2.score(train_expand, wine_train_label))
    #
    # sgd2 = SGDClassifier()
    # sgd2.fit(wine_train_features, wine_train_label)
    # show_result(' SGD unexpansion test', sgd2, wine_test_features, wine_test_label, CFmatrix=False)
    # print('SGD train accuracy is: ', sgd2.score(wine_train_features, wine_train_label))

    # ======================================================================== #
    #  <TEST> part 3: SGD best parameter CV

    param_grid_sgd = [
            {'max_iter': [100000000],
             'tol': [0.00000001, 0.0000005, 0.000001, 0.00005, 0.0001, 0.001, 0.005],
             'alpha': [ 0.0001,0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.05,0.1,1]
             }]
    grid_sgd = SGDClassifier()
    grid_search_sgd = GridSearchCV(grid_sgd, param_grid_sgd, cv=5, verbose=1)
    grid_search_sgd.fit(wine_train_features_st, wine_train_label)
    print(grid_search_sgd.cv_results_)
    sgd = grid_search_sgd.best_estimator_
    sgd.fit(train_expand_resample_st, train_label_resample)
    show_result('Model 4: SGDs', sgd, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('Model 4: SGD training accuracy', sgd.score(train_expand_resample_st, train_label_resample))
    print(grid_search_sgd.best_params_)
    print('SGD val accuracy', grid_search_sgd.best_score_)
    # best is: {'alpha': 0.001, 'max_iter': 100000000, 'tol': 5e-05}

    # ======================================================================== #
    # part 4: final best model on SGDClassifier test
    # oversample + feature expansion (degree=2) + Standardization/ alpha': 0.001, 'max_iter': 100000000, 'tol': 1e-06

    sgd_final = SGDClassifier(max_iter=1e+08, tol=1e-06, alpha=0.001)
    sgd_final.fit(train_expand_resample_st, train_label_resample)
    show_result('Model 4: SGD Classifier final testset', sgd_final, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('SGD Classifier train accuracy is: ', sgd_final.score(train_expand_resample_st, train_label_resample))

    '''Model 5: Random Forest Classifier'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # change the m to use different pre-processing method
    # rf_st = RandomForestClassifier()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     # m = StandardScaler()  # change this to compare the result of Standardization/Normalization for RF
    #     # m = MaxAbsScaler()
    #     m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)
    #     result, mean, std = CrossValidation(rf_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of RF is:', np.mean(mean_score))

    # # compare imbalance & balance
    # rf = RandomForestClassifier(random_state=0)
    # clf_rf = rf.fit(wine_train_features, wine_train_label)
    # show_result('Model 5: Random Forest test', rf, wine_test_features, wine_test_label, CFmatrix=False)
    # print('Random Forest training accuracy', rf.score(wine_train_features, wine_train_label))
    #
    # rf = RandomForestClassifier(random_state=0)
    # rf.fit(train_resample, train_label_resample)
    # show_result('Model 5: Random Forest test resample', rf, wine_test_features, wine_test_label, CFmatrix=False)
    # print('Random Forest training accuracy', rf.score(train_resample, train_label_resample))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion
    # to see the difference of feature expansion

    # maybe use expand
    # ply3 = PolynomialFeatures(degree=3)
    # tr3 = ply3.fit_transform(wine_train_features)
    # tt3 = ply3.transform(wine_test_features)
    #
    # rf2 = RandomForestClassifier()
    # rf2.fit(tr3, wine_train_label)
    # show_result(' RF 3 expansion test', rf2, tt3, wine_test_label, CFmatrix=False)
    # print('RF train accuracy is: ', rf2.score(tr3, wine_train_label))
    #
    # rf2 = RandomForestClassifier()
    # rf2.fit(train_expand, wine_train_label)
    # show_result(' RF expansion test', rf2, test_expand, wine_test_label, CFmatrix=False)
    # print('RF train accuracy is: ', rf2.score(train_expand, wine_train_label))
    #
    # rf2 = RandomForestClassifier()
    # rf2.fit(wine_train_features, wine_train_label)
    # show_result(' RF unexpansion test', rf2, wine_test_features, wine_test_label, CFmatrix=False)
    # print('RF train accuracy is: ', rf2.score(wine_train_features, wine_train_label))

    # ======================================================================== #
    #  <TEST> part 3: RF best parameter CV

    param_grid_rf = {'max_depth':range(5,25,2), 'n_estimators':range(15,101,10)}
    grid_rfc = RandomForestClassifier(random_state=0)
    grid_search_rf = GridSearchCV(grid_rfc, param_grid_rf, cv=5, verbose=1)
    grid_search_rf.fit(train_expand_resample_st, train_label_resample)
    print(grid_search_rf.cv_results_)
    rfc = grid_search_rf.best_estimator_
    rfc.fit(train_expand_resample_st, train_label_resample)

    show_result('Model 5: Random Forest', rfc, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print(grid_search_rf.best_params_)
    print('Random Forest train accuracy is: ', rfc.score(train_expand_resample_st, train_label_resample))
    print('RF val accuracy', grid_search_rf.best_score_)
    # best is: {'max_depth': 21, 'n_estimators': 150} random_state=0'max_depth': 21, 'n_estimators': 95, random_state=0

    # ======================================================================== #
    # part 4: final best model on Random Forest test
    # oversample + feature expansion (degree=2) + Standardization/ 0.62 best: 65,17 smote/ 0.616 n_estimators=150, max_depth=15, random_state=0

    rfc = RandomForestClassifier(n_estimators=150, max_depth=21, random_state=0) # 0.62 best: 65,17 smote/ 0.616 n_estimators=150, max_depth=15, random_state=0
    rfc.fit(train_expand_resample_st, train_label_resample)
    show_result('rf', rfc, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('best setting Radnom Forest train accuracy', rfc.score(train_expand_resample_st, train_label_resample))
    s = pickle.dumps(rfc)
    with open('model_WINE.pkl', 'wb+') as f:
        f.write(s)



    '''Model 6: MLP'''
    # ======================================================================== #
    # <TEST> Part 1 Pre-processing: get the performance of the Standardization/Normalization

    # change the m to use different pre-processing method
    # mlp_st = MLPClassifier()
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # cv_result = []
    # mean_score = []
    # std_score = []
    # for i in range(5):
    #     # m = StandardScaler()  # change this to compare the result of Standardization/Normalization for MLP
    #     # m = MaxAbsScaler()
    #     m = MinMaxScaler()
    #     data = m.fit_transform(wine_train_features)
    #     result, mean, std = CrossValidation(mlp_st, skf, data, wine_train_label)
    #     cv_result.append(result)
    #     mean_score.append(mean)
    #     std_score.append(std)
    # cv_result = np.array(cv_result)
    # mean_score = np.array(mean_score)
    # std_score = np.array(std_score)
    # print('==='*20)
    # print('Standardization/Normalization comparison validation accuracy of MLP is:', np.mean(mean_score))

    # compare imbalance & balance
    # mlp = MLPClassifier(random_state=0)
    # mlp.fit(wine_train_features, wine_train_label)
    # show_result('Model 6: MLP test', mlp, wine_test_features, wine_test_label, CFmatrix=False)
    # print('MLP training accuracy', mlp.score(wine_train_features, wine_train_label))
    #
    # mlp = MLPClassifier(random_state=0)
    # mlp.fit(train_resample, train_label_resample)
    # show_result('Model 6: MLP test resample', mlp, wine_test_features, wine_test_label, CFmatrix=False)
    # print('MLP training accuracy', mlp.score(train_resample, train_label_resample))

    # ======================================================================== #
    # <TEST> part 2: feature engineering - feature expansion
    # to see the difference of feature expansion

    # ply3 = PolynomialFeatures(degree=3)
    # tr3 = ply3.fit_transform(wine_train_features)
    # tt3 = ply3.transform(wine_test_features)
    #
    # mlp2 = MLPClassifier(random_state=0)
    # mlp2.fit(tr3, wine_train_label)
    # show_result(' MLP 3 expansion test', mlp2, tt3, wine_test_label, CFmatrix=False)
    # print('MLP train accuracy is: ', mlp2.score(tr3, wine_train_label))
    #
    # mlp2 = MLPClassifier(random_state=0)
    # mlp2.fit(train_expand, wine_train_label)
    # show_result(' MLP expansion test', mlp2, test_expand, wine_test_label, CFmatrix=False)
    # print('MLP train accuracy is: ', mlp2.score(train_expand, wine_train_label))
    #
    # mlp2 = MLPClassifier(random_state=0)
    # mlp2.fit(wine_train_features, wine_train_label)
    # show_result(' MLP unexpansion test', mlp2, wine_test_features, wine_test_label, CFmatrix=False)
    # print('MLP train accuracy is: ', mlp2.score(wine_train_features, wine_train_label))

    # ======================================================================== #
    #  <TEST> part 3: MLP best parameter CV

    param_grid_mlp = [
        {'hidden_layer_sizes': [10, 30, 50, 100, 150, 200],
         'activation': ["relu"],
         'solver': ["adam", "sgd"],
         'alpha': [0.0001, 0.001],
         'max_iter': [100000],
         'learning_rate_init': [0.0001, 0.001],
         'momentum': [0.95]
         }]
    grid_mlp = MLPClassifier()
    grid_search_mlp = GridSearchCV(grid_mlp, param_grid_mlp, cv=5, verbose=1)
    grid_search_mlp.fit(wine_train_features_st, wine_train_label)
    print(grid_search_mlp.cv_results_)
    mlp = grid_search_mlp.best_estimator_
    mlp.fit(wine_train_features_st, wine_train_label)
    show_result('Model 6: Multi-Layer Perceptron', mlp, wine_test_features_st, wine_test_label, CFmatrix=False)
    print('Multi-Layer Perceptron training accuracy', mlp.score(wine_train_features_st, wine_train_label))
    print(grid_search_mlp.best_params_)
    print('MLP val accuracy', grid_search_mlp.best_score_)

    # ======================================================================== #
    # part 4: final best model on MLP test
    # oversample + feature expansion (degree=2) + Standardization/

    mlp = MLPClassifier(hidden_layer_sizes=500,solver='adam',random_state=1, max_iter=100000, momentum=0.95, learning_rate_init=0.00001)  # 0.6 best: hidden_layer_sizes=600,random_state=1, max_iter=10000, momentum=0.9
    mlp.fit(train_expand_resample_st, train_label_resample)
    show_result('Model 6: Multi-Layer Perceptron', mlp, test_expand_resample_st, wine_test_label, CFmatrix=True)
    print('Multi-Layer Perceptron training accuracy', mlp.score(train_expand_resample_st, train_label_resample))


    plt.ioff()
    plt.show()

