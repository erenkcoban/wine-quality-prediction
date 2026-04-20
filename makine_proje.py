import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import os

# Streamlit App Title
st.set_page_config(page_title="Şarap Kalitesi Tahmin Uygulaması", layout="wide")
st.title("Şarap Kalitesi Tahmin Uygulaması")
st.write("""
Bu uygulama, kırmızı şarap örneklerinden alınan özellikler kullanılarak şarap kalitesini tahmin etmek için makine öğrenmesi modelleri kullanır.
Kullanıcı farklı veri işleme yöntemleri ve modeller arasında seçim yapabilir.
""")

# Sidebar for main options
st.sidebar.header("Model Seçimi ve İşlem Adımları")

# Veri Seti Yükleme - Kullanıcının veri seti seçmesine izin veriliyor
uploaded_file = st.sidebar.file_uploader("Bir CSV Dosyası Yükleyin", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(f"Yüklenen veri seti: *{os.path.basename(uploaded_file.name)}*")
else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")
    st.stop()

# Veri Seti Gösterme
if st.sidebar.checkbox("Veri Setini Göster"):
    st.write("Yüklenen veri setinin ilk 5 satırı:")
    st.write(data.head(10))

# Özellikler ve hedef değişkeni ayırma
if 'quality' in data.columns:
    X = data.drop(columns=['quality'])
    y = data['quality']
    st.write("'quality' sütunu hedef değişken olarak belirlendi.")
    st.write("Bu sütun, şarabın 0-10 arasında bir kalite skorunu temsil eder.")
else:
    st.error("'quality' sütunu bulunamadı. Lütfen uygun bir veri seti yükleyin.")
    st.stop()

# Kullanıcıdan model seçimi
model_choice = st.sidebar.selectbox("Model Seçimi", ["Logistic Regression", "Decision Tree", "Random Forest"])

# Gürültü Ekleme İşlemi
if st.sidebar.checkbox("Veriyi Gürültü Ekle"):
    noise_percentage = st.sidebar.slider("Gürültü Yüzdesi (%)", 0, 50, 10)
    num_noisy_points = int(len(X) * noise_percentage / 100)
    random_indices = np.random.choice(X.index, size=num_noisy_points, replace=False)

    noisy_data = X.copy()
    noisy_data.loc[random_indices] = noisy_data.loc[random_indices].apply(lambda x: x * np.random.uniform(0.5, 1.5), axis=1)

    st.sidebar.write(f"Veri setinin {noise_percentage}% oranında gürültü eklendi.")
    X = noisy_data

# Dengesiz Veri Seti İşlemi Seçimi
sampling_method = st.sidebar.selectbox("Dengesiz Veri Seti İşlemi", ["None", "Random Over-Sampling", "SMOTE"])

if sampling_method == "Random Over-Sampling":
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    st.sidebar.write("Random Over-Sampling ile veri dengelendi.")

elif sampling_method == "SMOTE":
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    st.sidebar.write("SMOTE ile veri dengelendi.")

# Eğitim ve test setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizasyon Seçimi
normalization_choice = st.sidebar.selectbox("Normalizasyon Türü", ["None", "MinMaxScaler", "StandardScaler"])

if normalization_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.sidebar.write("MinMaxScaler ile normalizasyon uygulandı.")

elif normalization_choice == "StandardScaler":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.sidebar.write("StandardScaler ile normalizasyon uygulandı.")

# Model Eğitimi
if model_choice == "Logistic Regression":
    model = LogisticRegression(random_state=42)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Performans Metrikleri
if st.sidebar.checkbox("Model Performansı"):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    st.subheader("Model Performans Metrikleri")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Doğruluk:", accuracy_score(y_test, y_pred))
        st.write("Hassasiyet:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Duyarlılık:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1-Skor:", f1_score(y_test, y_pred, average='weighted'))

    with col2:
        st.write("*Karışıklık Matrisi:*")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    st.subheader("Test Veri Setinde Predict ve Predict Proba")
    test_results = pd.DataFrame({"Gerçek Değer": y_test.values, "Tahmin": y_pred, "Tahmin Olasılıkları": [proba for proba in y_pred_proba]})
    st.write(test_results)

# K-Fold Cross Validation
if st.sidebar.checkbox("K-Fold Çapraz Doğrulama"):
    k = st.slider("K Değeri", 2, 5, 3)
    kf = KFold(n_splits=k)

    accuracies = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    st.write(f"{k}-Fold Çapraz Doğrulama Sonuçları")
    st.write("Doğruluklar:", accuracies)
    st.write("Ortalama Doğruluk:", accuracies.mean())

st.write("---")

# Kullanıcı Girdisi Alma
st.sidebar.header("Tahmin Girdisi Özellikleri ayarla")
user_input = []

if len(X.columns) > 0:
    for col in X.columns:
        value = st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        user_input.append(value)

# Tahmin Yapma Butonu
st.sidebar.markdown("---")
if len(user_input) == len(X.columns) and st.sidebar.button("Tahmin Yap"):
    user_input = np.array(user_input).reshape(1, -1)
    if normalization_choice != "None":
        user_input = scaler.transform(user_input)
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    st.subheader("Tahmin Sonuçları")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"*Tahmin Edilen Kalite*: {int(prediction[0])}")

    with col2:
        st.write("*predic proba Dağılımı:*")
        proba_df = pd.DataFrame(prediction_proba, columns=[f"Sınıf {cls}" for cls in model.classes_])
        st.write(proba_df)

st.write("Çalışma tamamlandı!")
