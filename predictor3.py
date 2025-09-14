import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6]=1
    preds[preds<.6]=0
    preds=pd.Series(preds,index=test.index,name="Predictions")
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

def backtest(data,model,predictors,start=2500,step=250):
    all_predictions=[]
    for i in range(start,data.shape[0],step):
        train=data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

sp500=yf.Ticker("^GSPC")
sp500=sp500.history(period="max")
plt.figure(figsize=(13,6))
sp500["Open"].plot.line(color="green")
sp500["Close"].plot.line(color="red",linestyle="--")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
sp500=sp500.drop(columns=["Dividends", "Stock Splits"])
sp500["Tomorrow"]=sp500["Close"].shift(-1)
sp500["Target"]=np.where(sp500["Tomorrow"]>sp500["Close"],1,0)
sp500=sp500.loc["1990-01-01":].copy()
print(sp500)
model=RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)
train=sp500.iloc[:-100]
test=sp500.iloc[-100:]
predictors=["Close","Volume","High","Low","Open"]
model.fit(train[predictors],train["Target"])
preds=model.predict(test[predictors])
preds=pd.Series(preds,index=test.index)
precisionscore=precision_score(test["Target"],preds)
print(precisionscore)
combined=pd.concat([test["Target"],preds],axis=1)
combined.plot()

plt.show()

horizons=[2,5,60,250,1000]
new_predictors=[]
for horizon in horizons:
    rolling_averages=sp500.rolling(horizon).mean()
    ratio_column=f"Close_Ratio_{horizon}"
    sp500[ratio_column]=sp500["Close"]/rolling_averages["Close"]
    trend_column=f"Trend_{horizon}"
    sp500[trend_column]=sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors+=[ratio_column,trend_column]
print(sp500)

predictions=backtest(sp500,model,new_predictors)
print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"],predictions["Predictions"]))

print(predictions["Target"].value_counts()/predictions.shape[0]) 