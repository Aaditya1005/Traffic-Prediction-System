import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

df["date_time"] = pd.to_datetime(df["date_time"])

df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day_of_week
df["month"] = df["date_time"].dt.month

def level(vol):
    if vol<0:
        return None
    elif vol<1193:
        return "Clear Road"
    elif vol<3380:
        return "Moderate"
    elif vol<4933:
        return "Moving Slowly"
    else:
        return "Heavy Congestion"

df["traffic_level"]=df["traffic_volume"].apply(level)
df = df.dropna(subset=["traffic_level"])

df = pd.get_dummies(df, columns=["weather_main", "holiday"])

features=["hour", "day", "month", "temp","rain_1h", "snow_1h", "clouds_all"]

X = df.drop(columns=["traffic_level", "traffic_volume", "date_time","weather_description"])
Y = df["traffic_level"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

model= RandomForestClassifier(random_state=42)

model.fit(X_train,Y_train)

Y_pred= model.predict(X_test)

accuracy= accuracy_score(Y_test,Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report: ")
print(classification_report(Y_test,Y_pred))

class_names = ["Clear Road", "Heavy Congestion", "Moderate", "Moving Slowly"]
matrix=confusion_matrix(Y_test,Y_pred,labels=class_names)


disp=ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=class_names)
disp.plot(cmap="Blues",xticks_rotation=45)

plt.title("Traffic Level Confusion Matrix")
plt.tight_layout()
plt.show()

joblib.dump(model,"traffic_model.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")
print("Model saved successfully!")
print("Feature columns saved successfully!")
