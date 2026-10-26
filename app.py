import streamlit as st
import pandas as pd
import joblib

st.title('Прогноз дощу в Австралії 🌧️')
st.write('Введіть дані про погоду сьогодні, і модель оцінить ймовірність дощу завтра.')

# Завантажуємо модель і всі об'єкти препроцесингу (кешуємо, щоб не читати файл щоразу)
@st.cache_resource
def load_artifacts():
    return joblib.load('aussie_rain.joblib')

art = load_artifacts()
model = art['model']
imputer = art['imputer']
scaler = art['scaler']
encoder = art['encoder']
numeric_cols = art['numeric_cols']
categorical_cols = art['categorical_cols']
encoded_cols = art['encoded_cols']
input_cols = art['input_cols']

# Списки значень для випадаючих меню беремо з енкодера (без nan)
categories = {}
for col, values in zip(categorical_cols, encoder.categories_):
    categories[col] = [v for v in values if pd.notna(v)]

# Розумні межі й типові значення для числових полів
# (мін, макс, типове значення)
ranges = {
    'MinTemp': (-10.0, 40.0, 12.0),
    'MaxTemp': (-5.0, 50.0, 23.0),
    'Rainfall': (0.0, 400.0, 0.0),
    'Evaporation': (0.0, 100.0, 5.0),
    'Sunshine': (0.0, 15.0, 8.0),
    'WindGustSpeed': (0.0, 140.0, 40.0),
    'WindSpeed9am': (0.0, 130.0, 14.0),
    'WindSpeed3pm': (0.0, 90.0, 19.0),
    'Humidity9am': (0.0, 100.0, 69.0),
    'Humidity3pm': (0.0, 100.0, 51.0),
    'Pressure9am': (980.0, 1042.0, 1017.0),
    'Pressure3pm': (977.0, 1040.0, 1015.0),
    'Cloud9am': (0.0, 9.0, 4.0),
    'Cloud3pm': (0.0, 9.0, 5.0),
    'Temp9am': (-8.0, 45.0, 17.0),
    'Temp3pm': (-5.0, 50.0, 21.0),
}

# Збираємо введення користувача в словник
user_input = {}

st.subheader('Місце і напрямки вітру')
col1, col2 = st.columns(2)
user_input['Location'] = col1.selectbox('Місто (метеостанція)', categories['Location'])
user_input['RainToday'] = col2.selectbox('Чи був дощ сьогодні?', categories['RainToday'])
user_input['WindGustDir'] = col1.selectbox('Напрямок поривів вітру', categories['WindGustDir'])
user_input['WindDir9am'] = col2.selectbox('Напрямок вітру о 9:00', categories['WindDir9am'])
user_input['WindDir3pm'] = col1.selectbox('Напрямок вітру о 15:00', categories['WindDir3pm'])

st.subheader('Числові показники')
cols = st.columns(2)
for i, name in enumerate(numeric_cols):
    low, high, default = ranges[name]
    user_input[name] = cols[i % 2].number_input(name, min_value=low, max_value=high, value=default)

# Функція прогнозу: той самий препроцесинг, що й при навчанні
def predict_rain(single_input):
    df = pd.DataFrame([single_input])[input_cols]
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    X = df[numeric_cols + encoded_cols]
    prob_yes = model.predict_proba(X)[0][list(model.classes_).index('Yes')]
    return prob_yes

if st.button('Зробити прогноз'):
    prob_yes = predict_rain(user_input)

    # Прогноз: Yes, якщо ймовірність дощу більша за 50%
    prediction = 'Yes' if prob_yes > 0.5 else 'No'
    # Ймовірність того класу, який ми прогнозуємо
    prob_prediction = prob_yes if prediction == 'Yes' else 1 - prob_yes

    st.subheader(f'Прогноз: {prediction}')
    st.write(f'Ймовірність цього прогнозу: **{prob_prediction * 100:.1f} %**')
    st.write(f'Ймовірність дощу завтра: {prob_yes * 100:.1f} %')

    if prediction == 'Yes':
        st.error('Завтра, ймовірно, буде дощ ☔')
    else:
        st.success('Завтра, ймовірно, дощу не буде ☀️')
