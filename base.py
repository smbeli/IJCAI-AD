import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 读取数据
train = pd.read_csv("input/train.txt", sep="\s+")
test = pd.read_csv("input/test.txt", sep="\s+")
#data = pd.concat([train, test])

# 选择特征
select_cols = ['item_price_level','item_sales_level','item_collected_level','item_pv_level','user_gender_id','user_age_level','user_star_level','context_page_id',
              'shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']

def model_log_loss(model):
    X = train[select_cols]
    Y = train['is_trade']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4, random_state=0)

    print("Training...")
    model.fit(X_train, y_train)
    print("Predicting...")
    y_prediction = model.predict_proba(X_test)
    test_pred = y_prediction[:, 1]
    print('log_loss ', log_loss(y_test, test_pred))

def result(model):
    X = train[select_cols]
    Y = train['is_trade']
    model.fit(X,Y)
    y_pred = model.predict_proba(test[select_cols])[:,1]
    result = pd.DataFrame({'instance_id':test['instance_id'], 'predicted_score':y_pred})
    result.to_csv('result/result0411.txt',sep=" ",index=False)

def lgb_log_loss():
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
    X = train[select_cols]
    Y = train['is_trade']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4, random_state=0)
    print("Training...")
    lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=200)
    print("Predicting...")
    y_prediction = lgb_model.predict_proba(X_test)
    test_pred = y_prediction[:, 1]
    print('log_loss ', log_loss(y_test, test_pred))

if __name__ == "__main__":
    #lgb_log_loss()
    #print(test['instance_id'].values)
    result(LogisticRegression(C=100,n_jobs=-1))
    #print("---LogisticRegression------")
    #model_log_loss(LogisticRegression(C=100,n_jobs=-1))
    # print("---DecisionTreeClassifier---")
    # model_log_loss(DecisionTreeClassifier())
    # print("---RandomForestClassifier---")
    # model_log_loss(RandomForestClassifier())
    #print("---SVC---")
    #model_log_loss(SVC())

    