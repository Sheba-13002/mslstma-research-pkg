\# MSLSTMA Stock Price Prediction Engine



Proprietary stock price prediction pipeline utilizing Refinitiv Data and a compiled LSTM-based neural network architecture.



\## Requirements

\- Python (in `refinitiv\_env` Conda environment)

\- Refinitiv Data (`refinitiv-data`)

\- TensorFlow / Keras

\- Pandas, NumPy, Scikit-learn, Matplotlib



\## Quick Start



Run the training and prediction pipeline:

```bash

python main.py --ticker INTC.O --epochs 1000 --batch_size 16
