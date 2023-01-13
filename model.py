from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

_solver = 'lbfgs'
_activation = 'relu'
_alpha = 0.001
_hidden_layer_nodes = (32, 6, 5)
_random_state = 3
_max_iter = 6000
_warm_start = True

def find_best_parameters(feature_train, target_train):
    parameters = {
        'solver': ('lbfgs', 'adam'),
        'hidden_layer_sizes': ((32, 16, 8, 4, 2), (64, 32, 16, 8, 4), (128, 64, 32, 16, 8)),
        'alpha': (0.001, 0.0001),
        'tol': (0.001, 0.0001),
        'max_iter': (20000, 25000),
        'max_fun': (20000, 25000)
    }
    MLP = MLPRegressor()
    clf = GridSearchCV(MLP, parameters, scoring='neg_mean_squared_error', cv=10)
    clf.fit(feature_train, target_train)
    sorted(clf.cv_results_.keys())

def MLP_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    MLP_model = MLPRegressor(solver=_solver, alpha=_alpha, hidden_layer_sizes=_hidden_layer_nodes,
                             random_state=_random_state, max_iter=_max_iter, activation=_activation,
                             warm_start=_warm_start)
    trained_model = MLP_model.fit(feature_train, target_train)
    R2_train_score = MLP_model.score(feature_train, target_train)
    print("MLP R2 Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    print("Cross validation average RMSE scores: " + str(Average_RMSE_score))
    cross_val_MAE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    print("Cross validation average MAE scores: " + str(Average_MAE_score))
    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)
    R2_test_score = MLP_model.score(feature_test, target_test)
    print("MLP R2 test Scores:" + str(R2_test_score))
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))
    # print(test_predict)
    # print(target_test)

    return MLP_model.predict, train_predict, test_predict, trained_model

def RF_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    RF_model = RandomForestRegressor(max_depth=11, n_estimators=30, max_features=6, min_samples_leaf=3, min_samples_split=4)
    trained_model = RF_model.fit(feature_train, target_train)
    R2_train_score = RF_model.score(feature_train, target_train)
    print("RF R2 Scores: " + str(R2_train_score))
    cross_val_RMSE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    Average_RMSE_score = cross_val_RMSE.mean()
    print("Cross validation average RMSE scores: " + str(Average_RMSE_score))
    cross_val_MAE = -1 * cross_val_score(RF_model, feature_train, target_train, cv=kf, scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    print("Cross validation average MAE scores: " + str(Average_MAE_score))
    R2_test_score = RF_model.score(feature_test, target_test)
    print("RF R2 test Scores:" + str(R2_test_score))
    train_predict = RF_model.predict(feature_train)
    test_predict = RF_model.predict(feature_test)
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))

    return RF_model.predict, train_predict, test_predict

def SVR_predict(feature_train, target_train, feature_test, target_test):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    SVR_model = SVR(kernel='linear', C=1, epsilon=0.001)
    trained_model = SVR_model.fit(feature_train, target_train)
    R2_train_score = SVR_model.score(feature_train, target_train)
    print("SVR R2 Scores: " + str(R2_train_score))
    RMSE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    Average_MAE_score = RMSE.mean()
    print("Average RMSE scores: " + str(Average_MAE_score))
    cross_val_MAE = -1 * cross_val_score(SVR_model, feature_train, target_train, cv=kf, scoring='neg_mean_absolute_error')
    Average_MAE_score = cross_val_MAE.mean()
    print("Cross validation average MAE scores: " + str(Average_MAE_score))
    R2_test_score = SVR_model.score(feature_test, target_test)
    print("SVR R2 test Scores:" + str(R2_test_score))
    train_predict = SVR_model.predict(feature_train)
    test_predict = SVR_model.predict(feature_test)
    test_MAE = mean_absolute_error(target_test, test_predict)
    print("Test data MAE:" + str(test_MAE))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    print("Test data RMSE:" + str(test_RMSE))

    return SVR_model.predict, train_predict, test_predict

def MLP_auto(feature_train, target_train, feature_test, target_test, hidden_layer_param):
    kf = KFold(n_splits=10, shuffle=True, random_state=5)
    MLP_model = MLPRegressor(solver=_solver, alpha=_alpha, hidden_layer_sizes=hidden_layer_param,
                             random_state=_random_state, max_iter=_max_iter, activation=_activation,
                             warm_start=_warm_start)
    trained_model = MLP_model.fit(feature_train, target_train)
    R2_train_score = MLP_model.score(feature_train, target_train)
    # print("MLP R2 Scores: " + str(R2_train_score))
    train_predict = MLP_model.predict(feature_train)
    test_predict = MLP_model.predict(feature_test)
    R2_test_score = MLP_model.score(feature_test, target_test)
    # print("MLP R2 test Scores:" + str(R2_test_score))
    test_MAE = mean_absolute_error(target_test, test_predict)
    # print("Test data MAE:" + str(test_MAE))
    test_RMSE = mean_squared_error(target_test, test_predict) ** 0.5
    # print("Test data RMSE:" + str(test_RMSE))
    # cross_val_RMSE = -1 * cross_val_score(MLP_model, feature_train, target_train, cv=kf, scoring='neg_root_mean_squared_error')
    # Average_RMSE_score = cross_val_RMSE.mean()

    return R2_test_score
