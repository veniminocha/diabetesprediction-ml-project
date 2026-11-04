import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Predictor")
st.caption("Predictive Analytics mini-project | Educational demonstration only — not a medical diagnosis.")

FEATURES=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
ZERO_MISSING=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
URL="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

@st.cache_data
def load_data():
    df=pd.read_csv(URL,header=None,names=FEATURES+["Outcome"])
    return df

@st.cache_resource
def build_models(df):
    X=df[FEATURES].copy()
    y=df["Outcome"].astype(int)
    for c in ZERO_MISSING:
        X.loc[X[c]==0,c]=np.nan
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
        "Logistic Regression":LogisticRegression(max_iter=2000),
        "LDA":LinearDiscriminantAnalysis(),
        "QDA":QuadraticDiscriminantAnalysis(reg_param=.1)
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]; fitted={}
    for name,model in models.items():
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",model)])
        scores=cross_validate(pipe,Xtr,ytr,cv=cv,scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"})
        rows.append({"Model":name,"CV Accuracy":scores["test_accuracy"].mean(),"CV Precision":scores["test_precision"].mean(),"CV Recall":scores["test_recall"].mean(),"CV F1":scores["test_f1"].mean(),"CV ROC-AUC":scores["test_roc_auc"].mean()})
        pipe.fit(Xtr,ytr); fitted[name]=pipe
    table=pd.DataFrame(rows).sort_values("CV ROC-AUC",ascending=False)
    best=table.iloc[0]["Model"]; model=fitted[best]
    pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    metrics={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,prob)}
    return X,y,Xte,yte,table,best,model,pred,prob,metrics

with st.sidebar:
    st.header("Project")
    st.write("Dataset: Pima Indians Diabetes")
    st.write("768 records • 8 input features")
    st.info("For educational use only. Not for healthcare decisions.")

try:
    df=load_data()
    X,y,Xte,yte,comparison,best,model,pred,prob,metrics=build_models(df)
except Exception as e:
    st.error("Could not load/train the project. Check your internet connection and try again.")
    st.exception(e)
    st.stop()

t1,t2,t3,t4=st.tabs(["Dashboard","Explore Data","Try Prediction","About"])
with t1:
    st.subheader("Overview")
    a,b,c,d=st.columns(4)
    a.metric("Records",len(df)); b.metric("Features",8); c.metric("Selected model",best); d.metric("Test accuracy",f"{metrics['Accuracy']:.1%}")
    a,b,c=st.columns(3)
    a.metric("Precision",f"{metrics['Precision']:.1%}"); b.metric("Recall",f"{metrics['Recall']:.1%}"); c.metric("Test ROC-AUC",f"{metrics['ROC-AUC']:.3f}")
    st.markdown("### 5-fold cross-validation comparison")
    st.dataframe(comparison.style.format({col:"{:.3f}" for col in comparison.columns if col!="Model"}),use_container_width=True)
    fig,ax=plt.subplots(figsize=(7,3.5)); sns.barplot(data=comparison,x="Model",y="CV ROC-AUC",ax=ax); ax.set_ylim(0,1); st.pyplot(fig); plt.close(fig)
    left,right=st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        fig,ax=plt.subplots(); sns.heatmap(confusion_matrix(yte,pred),annot=True,fmt="d",cmap="Blues",ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("### ROC curve")
        fpr,tpr,_=roc_curve(yte,prob); fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f"AUC={metrics['ROC-AUC']:.3f}"); ax.plot([0,1],[0,1],"--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(); st.pyplot(fig); plt.close(fig)
with t2:
    st.subheader("Explore dataset")
    st.dataframe(df.head(10),use_container_width=True)
    l,r=st.columns(2)
    with l:
        st.markdown("### Outcome distribution")
        counts=df.Outcome.value_counts().sort_index(); fig,ax=plt.subplots(); ax.bar(["Class 0","Class 1"],counts.values); ax.set_ylabel("Records"); st.pyplot(fig); plt.close(fig)
    with r:
        feat=st.selectbox("Feature",FEATURES,index=1); fig,ax=plt.subplots(); sns.histplot(data=df,x=feat,hue="Outcome",bins=25,kde=True,ax=ax); st.pyplot(fig); plt.close(fig)
    st.markdown("### Feature correlations")
    fig,ax=plt.subplots(figsize=(9,5)); sns.heatmap(df.corr(numeric_only=True),annot=True,fmt=".2f",cmap="coolwarm",ax=ax); st.pyplot(fig); plt.close(fig)
with t3:
    st.subheader("Try a sample prediction")
    st.write("Enter example measurements. The output is a model classification, not a clinical assessment.")
    with st.form("predict"):
        l,r=st.columns(2)
        with l:
            pregnancies=st.number_input("Pregnancies",0,20,1)
            glucose=st.number_input("Glucose (mg/dL)",0.0,300.0,120.0)
            bp=st.number_input("Blood pressure (mmHg)",0.0,200.0,70.0)
            skin=st.number_input("Skin thickness (mm)",0.0,100.0,20.0)
        with r:
            insulin=st.number_input("Insulin",0.0,900.0,80.0)
            bmi=st.number_input("BMI",0.0,80.0,28.0)
            dpf=st.number_input("Diabetes pedigree function",0.0,3.0,.5,step=.01)
            age=st.number_input("Age",1,120,30)
        go=st.form_submit_button("Predict",type="primary")
    if go:
        row=pd.DataFrame([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]],columns=FEATURES)
        for col in ZERO_MISSING:
            row.loc[row[col]==0,col]=np.nan
        p=float(model.predict_proba(row)[0,1]); label=int(p>=.5)
        st.metric("Estimated model probability for class 1",f"{p:.1%}")
        if label: st.warning("Model classifies this example as class 1.")
        else: st.success("Model classifies this example as class 0.")
        st.caption("Not a diagnosis or substitute for professional medical advice.")
with t4:
    st.markdown("""
### Workflow
1. Load the Pima Indians Diabetes dataset.
2. Treat zero values in selected clinical measurements as missing.
3. Impute using median and scale features inside a pipeline.
4. Split into stratified training and test sets.
5. Compare Logistic Regression, LDA, and QDA using 5-fold cross-validation.
6. Select the highest mean CV ROC-AUC model and evaluate on the held-out test set.

### Limitations
The dataset is small and population-specific. Results may not generalize to other populations. This project is for educational demonstration only.
""")
    st.download_button("Download dataset CSV",df.to_csv(index=False),"diabetes_dataset.csv","text/csv")
