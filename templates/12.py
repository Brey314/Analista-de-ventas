from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import sqrt
import io
import json
from datetime import datetime, timedelta

app = Flask(__name__)

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Obtener archivo subido
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        # Leer archivo
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Formato de archivo no soportado'}), 400
        
        # Procesar parámetros
        params = json.loads(request.form['params'])
        date_col = params['dateColumn']
        value_col = params['valueColumn']
        forecast_periods = int(params['forecastPeriods'])
        model_type = params['modelType']
        
        # Preparar datos
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        y = df[value_col]
        
        # División en train-test (80-20)
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = {
            'dates': y_test.index.strftime('%Y-%m-%d').tolist(),
            'actual': y_test.values.tolist(),
            'weekly_data': df.reset_index().rename(columns={
                date_col: 'Fecha',
                value_col: 'Venta'
            }).to_dict('records')
        }
        
        # Ejecutar modelos según selección
        if model_type == 'auto' or model_type == 'arima':
            # Modelo ARIMA
            auto_model = auto_arima(y_train, seasonal=False, trace=False)
            model_arima = ARIMA(y_train, order=auto_model.order)
            results_arima = model_arima.fit()
            forecast_arima = results_arima.get_forecast(steps=len(y_test))
            arima_pred = forecast_arima.predicted_mean
            
            results['arima'] = arima_pred.tolist()
            results['metrics'] = {
                'arima_mae': mean_absolute_error(y_test, arima_pred),
                'arima_rmse': sqrt(mean_squared_error(y_test, arima_pred))
            }
        
        if model_type == 'auto' or model_type == 'sarima':
            # Modelo SARIMAX (simplificado para el ejemplo)
            seasonal_model = auto_arima(y_train, seasonal=True, m=4, trace=False)
            model_sarima = SARIMAX(y_train, order=seasonal_model.order,
                                 seasonal_order=seasonal_model.seasonal_order)
            results_sarima = model_sarima.fit(disp=False)
            forecast_sarima = results_sarima.get_forecast(steps=len(y_test))
            sarima_pred = forecast_sarima.predicted_mean
            
            results['sarima'] = sarima_pred.tolist()
            if 'metrics' not in results:
                results['metrics'] = {}
            results['metrics']['sarima_mae'] = mean_absolute_error(y_test, sarima_pred)
            results['metrics']['sarima_rmse'] = sqrt(mean_squared_error(y_test, sarima_pred))
        
        if model_type == 'auto' or model_type == 'lstm':
            # Modelo LSTM
            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
            
            n_steps = 4
            X_train, y_train_lstm = create_sequences(y_scaled, n_steps)
            
            model_lstm = Sequential([
                LSTM(50, activation='relu', input_shape=(n_steps, 1)),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_train, y_train_lstm, epochs=200, verbose=0)
            
            test_scaled = scaler.transform(y.values.reshape(-1, 1))
            X_test, y_test_lstm = create_sequences(test_scaled, n_steps)
            X_test = X_test[-(len(y_test)-n_steps):]
            
            lstm_pred_scaled = model_lstm.predict(X_test)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
            
            # Ajustar fechas para LSTM (por los n_steps)
            lstm_dates = y_test.index[n_steps:].strftime('%Y-%m-%d').tolist()
            
            results['lstm'] = lstm_pred.flatten().tolist()
            results['lstm_dates'] = lstm_dates
            if 'metrics' not in results:
                results['metrics'] = {}
            results['metrics']['lstm_mae'] = mean_absolute_error(y_test[n_steps:], lstm_pred)
            results['metrics']['lstm_rmse'] = sqrt(mean_squared_error(y_test[n_steps:], lstm_pred))
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample-data', methods=['GET'])
def sample_data():
    # Datos de ejemplo predefinidos
    data = {
        'Semana': [f"Semana {i}" for i in range(1, 35)],
        'CANTIDAD': [
            34, 37, 41, 74, 119, 92, 144, 92, 81, 115,
            45, 66, 120, 20, 75, 88, 94, 145, 163, 113,
            81, 225, 138, 277, 108, 211, 119, 95, 115, 90,
            126, 130, 85, 89
        ],
        'Venta': [
            82700, 87100, 84200, 189100, 443300, 221600, 326200, 329500, 242200, 99800,
            143800, 196000, 433500, 64000, 185000, 212000, 235000, 490700, 535100, 386000,
            294000, 621500, 528000, 963500, 402000, 801000, 467500, 371500, 460000, 278000,
            399900, 393100, 230000, 245000
        ],
        'Fecha': pd.date_range(start='2023-01-02', periods=34, freq='W-Mon').strftime('%Y-%m-%d').tolist()
    }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)