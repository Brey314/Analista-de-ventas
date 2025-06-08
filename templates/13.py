import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import sqrt
import matplotlib.dates as mdates

# 1. Crear DataFrame con los datos semanales
def create_weekly_data():
    data = {
        'Semana': [f"Semana {i}" for i in range(1, 35)],
        'CANTIDAD': [
            34, 37, 41, 74, 119, 92, 144, 92, 81, 115,
            45, 66, 120, 20, 1, 9, 5, 145, 163, 113,
            81, 225, 138, 277, 108, 211, 119, 95, 115, 11,
            126, 130, 4, 9
        ],
        'Venta': [
            82700, 87100, 84200, 189100, 443300, 221600, 326200, 329500, 242200, 99800,
            143800, 196000, 433500, 64000, 20000, 19800, 15000, 490700, 535100, 386000,
            294000, 621500, 528000, 963500, 402000, 801000, 467500, 371500, 460000, 38500,
            399900, 393100, 16000, 19800
        ]
    }
    
    df = pd.DataFrame(data)
    df['Fecha'] = pd.date_range(start='2023-01-02', periods=len(df), freq='W-Mon')
    df = df.set_index('Fecha')
    
    return df

# Crear datos semanales
df = create_weekly_data()
print("Datos semanales creados:")
print(df.head())

# 2. Preparar datos para modelos
y = df['Venta']  # Usamos ventas semanales como variable objetivo

# División en train-test (80-20)
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

# ----------------------------------------------------------
# MODELO ARIMA
# ----------------------------------------------------------
print("\nENTRENANDO MODELO ARIMA...")
try:
    auto_model = auto_arima(y_train, seasonal=False, trace=True, suppress_warnings=True)
    print(f"Mejores parámetros ARIMA: {auto_model.order}")

    model_arima = ARIMA(y_train, order=auto_model.order)
    results_arima = model_arima.fit()
    forecast_arima = results_arima.get_forecast(steps=len(y_test))
    arima_pred = forecast_arima.predicted_mean

    print(f"MAE ARIMA: {mean_absolute_error(y_test, arima_pred):,.2f}")
    print(f"RMSE ARIMA: {sqrt(mean_squared_error(y_test, arima_pred)):,.2f}")
except Exception as e:
    print(f"Error en ARIMA: {str(e)}")
    arima_pred = np.zeros(len(y_test))  # Predicción dummy en caso de error

# ----------------------------------------------------------
# MODELO SARIMAX (usando cantidad como variable exógena)
# ----------------------------------------------------------
print("\nENTRENANDO MODELO SARIMAX...")
try:
    exog = df[['CANTIDAD']]  # Usamos cantidad semanal como variable exógena
    exog_train, exog_test = exog[:split_idx], exog[split_idx:]

    seasonal_model = auto_arima(y_train, exogenous=exog_train, seasonal=True, m=4,
                              trace=True, suppress_warnings=True)
    print(f"Mejores parámetros SARIMAX: {seasonal_model.order} {seasonal_model.seasonal_order}")

    model_sarima = SARIMAX(y_train, exog=exog_train, 
                          order=seasonal_model.order,
                          seasonal_order=seasonal_model.seasonal_order)
    results_sarima = model_sarima.fit(disp=False)
    forecast_sarima = results_sarima.get_forecast(steps=len(y_test), exog=exog_test)
    sarima_pred = forecast_sarima.predicted_mean

    print(f"MAE SARIMAX: {mean_absolute_error(y_test, sarima_pred):,.2f}")
    print(f"RMSE SARIMAX: {sqrt(mean_squared_error(y_test, sarima_pred)):,.2f}")
except Exception as e:
    print(f"Error en SARIMAX: {str(e)}")
    sarima_pred = np.zeros(len(y_test))  # Predicción dummy en caso de error

# ----------------------------------------------------------
# MODELO LSTM
# ----------------------------------------------------------
print("\nENTRENANDO MODELO LSTM...")
try:
    # Escalado
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data)-n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = 4  # Usar 4 semanas como historial
    X_train, y_train_lstm = create_sequences(y_scaled, n_steps)

    model_lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')

    history = model_lstm.fit(
        X_train, 
        y_train_lstm, 
        epochs=200, 
        verbose=0,
        validation_split=0.2
    )

    # Preparar test
    test_scaled = scaler.transform(y.values.reshape(-1, 1))
    X_test, y_test_lstm = create_sequences(test_scaled, n_steps)
    X_test = X_test[-(len(y_test)-n_steps):]

    lstm_pred_scaled = model_lstm.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    y_test_lstm_adj = y_test[n_steps:]

    print(f"MAE LSTM: {mean_absolute_error(y_test_lstm_adj, lstm_pred):,.2f}")
    print(f"RMSE LSTM: {sqrt(mean_squared_error(y_test_lstm_adj, lstm_pred)):,.2f}")
except Exception as e:
    print(f"Error en LSTM: {str(e)}")
    lstm_pred = np.zeros(len(y_test)-n_steps)  # Predicción dummy en caso de error
    y_test_lstm_adj = y_test[n_steps:]

# ----------------------------------------------------------
# GRÁFICOS MEJORADOS
# ----------------------------------------------------------

# 1. Gráfico de serie temporal con formato de dinero
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Venta'], 'b-', label='Ventas Reales')
plt.title('Ventas Semanales', fontsize=16, pad=20)
plt.xlabel('Fecha', fontsize=12, labelpad=10)
plt.ylabel('Ventas ($)', fontsize=12, labelpad=10)

# Formatear eje Y como dinero
plt.gca().yaxis.set_major_formatter('${x:,.0f}')

# Formatear fechas en el eje X
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# 2. Gráfico de dispersión Cantidad vs Ventas
plt.figure(figsize=(10, 6))
plt.scatter(df['CANTIDAD'], df['Venta'], c='#f72585', alpha=0.7)
plt.title('Relación entre Cantidad y Ventas', fontsize=16, pad=20)
plt.xlabel('Cantidad Vendida', fontsize=12, labelpad=10)
plt.ylabel('Ventas ($)', fontsize=12, labelpad=10)

# Formatear eje Y como dinero
plt.gca().yaxis.set_major_formatter('${x:,.0f}')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Comparación de Modelos con formato mejorado
plt.figure(figsize=(15, 8))

# Ajustar índices para LSTM
test_dates = y_test.index
lstm_dates = y_test.index[n_steps:]

plt.plot(test_dates, y_test, 'k-', label='Real', linewidth=3)
plt.plot(test_dates, arima_pred, 'r--', label='ARIMA', linewidth=2)
plt.plot(test_dates, sarima_pred, 'g--', label='SARIMAX', linewidth=2)
plt.plot(lstm_dates, lstm_pred, 'm--', label='LSTM', linewidth=2)

plt.title('Comparación de Modelos - Ventas Semanales', fontsize=16, pad=20)
plt.xlabel('Fecha', fontsize=12, labelpad=10)
plt.ylabel('Venta Semanal ($)', fontsize=12, labelpad=10)

# Formatear eje Y como dinero
plt.gca().yaxis.set_major_formatter('${x:,.0f}')

# Formatear fechas en el eje X
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.xticks(rotation=45)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Gráfico de barras de las ventas por semana
plt.figure(figsize=(15, 6))
plt.bar(df.index, df['Venta'], color='#4361ee', width=5)
plt.title('Ventas Semanales', fontsize=16, pad=20)
plt.xlabel('Semana', fontsize=12, labelpad=10)
plt.ylabel('Ventas ($)', fontsize=12, labelpad=10)

# Formatear eje Y como dinero
plt.gca().yaxis.set_major_formatter('${x:,.0f}')

# Formatear fechas en el eje X
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.xticks(rotation=45)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Mostrar tabla de datos semanales
print("\nTabla de datos semanales utilizada:")
print(df[['Semana', 'CANTIDAD', 'Venta']])