import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)

df = pd.read_csv('LKOH.csv', delimiter=';')

model = Prophet()
model.fit(df[:1000])

# i, j = 240 // 2, 241 // 2
# median_def = np.sort(df.y[1000:])
# median_def = np.mean([median_def[i], median_def[j]])

prediction = model.make_future_dataframe(periods=360)
prediction_df = model.predict(prediction)

# i = len(prediction_df) // 2
# median_ml = np.sort(prediction_df.yhat)
# median_ml = np.mean([median_ml[i]])

plt.subplot(1, 1, 1)
plt.plot(prediction_df.ds, prediction_df.yhat_lower, color='yellow', label='нижнее значение (yhat_lower)')
plt.plot(prediction_df.ds, prediction_df.yhat, color='orange', label='среднее значение (yhat)')
plt.plot(prediction_df.ds, prediction_df.yhat_upper, color='red', label='верхнее значение (yhat_upper)')
plt.legend(loc='upper left')

figure_prediction = model.plot(prediction_df)

cross_validation_df = cross_validation(model, initial='1080 days', period='180 days', horizon='360 days')
performance = performance_metrics(cross_validation_df)

figure_approximate = plot_cross_validation_metric(cross_validation_df, metric='mape')
print('MAE =', np.mean(performance['mae']), '\n',
      'MAPE =', np.mean(performance['mape']), '\n')
#       'Изначальная медиана:', median_def, '\nСпрогнозированная медиана:', np.round(median_ml, 2))

plt.show()
