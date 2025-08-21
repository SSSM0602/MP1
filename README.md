# Layer Prediction Web Application

A Flask-based web application that uses machine learning to predict layer classifications based on test measurements. The application features a Multi-Layer Perceptron (MLP) neural network for prediction and includes functionality for model training, prediction, and data storage.

## Features

- **Model Training**: Train an MLP neural network using data stored in SQLite database
- **Prediction Interface**: Web form for inputting test values and getting layer predictions
- **Data Persistence**: Automatic storage of predictions in SQLite database
- **Model Performance**: Display training accuracy metrics
- **Pipeline Architecture**: Integrated preprocessing with StandardScaler and MLP classification

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn
  - Multi-Layer Perceptron (MLPClassifier)
  - StandardScaler for feature preprocessing
  - Pipeline for streamlined processing
- **Database**: SQLite
- **Data Processing**: pandas
- **Model Persistence**: joblib

## Project Structure

```
layering-prediction-app/
├── app.py                 # Main Flask application
├── layering.db           # SQLite database (created automatically)
├── model.pkl             # Trained model (created after training)
├── templates/
│   ├── index.html        # Home page
│   ├── train.html        # Model training interface
│   ├── accuracy.html     # Training results display
│   ├── predict.html      # Prediction input form
│   └── prediction.html   # Prediction results display
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd layering-prediction-app
```

2. Install required dependencies:
```bash
pip install flask sqlite3 pandas scikit-learn joblib
```

## Usage

### Starting the Application

```bash
python app.py
```

The application will run on `http://localhost:5000` in debug mode.

### Training the Model

1. Navigate to `/train` route
2. Ensure your SQLite database contains a `circles` table with columns:
   - `Test1` (numerical feature)
   - `Test2` (numerical feature) 
   - `layer` (target classification)
3. Click the training button to:
   - Load data from database
   - Clean and preprocess the data
   - Split into training/testing sets (80/20)
   - Train MLP model with architecture: 100→50 hidden layers
   - Save the trained model as `model.pkl`
   - Display training accuracy

### Making Predictions

1. Navigate to `/predict` route
2. Enter values for Feature 1 and Feature 2
3. Submit to get layer prediction
4. Prediction results are automatically stored in the `predictions` table

## Database Schema

### circles table (training data)
- `Test1`: REAL - First test measurement
- `Test2`: REAL - Second test measurement  
- `layer`: INTEGER - Layer classification (target variable)

### predictions table (prediction history)
- `id`: INTEGER PRIMARY KEY
- `feature1`: REAL - First input feature
- `feature2`: REAL - Second input feature
- `prediction`: INTEGER - Model prediction result

## Model Details

- **Algorithm**: Multi-Layer Perceptron (MLP) Neural Network
- **Architecture**: Input → 100 neurons → 50 neurons → Output
- **Activation**: ReLU
- **Preprocessing**: StandardScaler for feature normalization
- **Train/Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)

## API Routes

- `GET /` - Home page
- `GET/POST /train` - Model training interface
- `GET/POST /predict` - Prediction interface

## Future Enhancements

- Add data visualization for training data and predictions
- Implement model evaluation metrics beyond accuracy
- Add data upload functionality for training data
- Include model versioning and comparison features
- Add user authentication and session management


## License

[Add your preferred license here]
