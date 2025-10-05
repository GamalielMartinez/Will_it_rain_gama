# Rain Prediction System

A machine learning-based system for predicting precipitation probability in the Gulf of Mexico region using HDF5 meteorological data.

## Overview

This system uses a Random Forest regression model trained on GPM IMERG satellite precipitation data to predict rainfall probability at specific geographic coordinates and times.

## Features

- **Web Interface**: Interactive map-based interface using NiceGUI and Folium
- **Console Interface**: Command-line tool for quick predictions
- **Time Selection**: Predict rainfall for specific hours and minutes
- **Geographic Coverage**: Optimized for Gulf of Mexico coastal regions
- **Exact Probabilities**: Returns precise percentage probabilities instead of categorical ranges

## Technical Specifications

### Model Details
- **Algorithm**: Random Forest Regression
- **Training Data**: 212 HDF5 GPM IMERG files
- **Features**: 57 engineered features including temporal, geographic, and meteorological variables
- **Performance**: R² score of 0.8483 (Out-of-Bag)

### Feature Engineering
- Temporal features (hour, day, month with cyclical encoding)
- Geographic features (latitude, longitude, coastal factors)
- Statistical aggregations (mean, median, percentiles)
- Precipitation patterns and trends
- Quality and error metrics

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd will_it_rain_gama
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the trained model file `rain_prediction_model.joblib` is present in the root directory.

## Usage

### Web Interface
```bash
python src/main.py
```
Access the application at `http://localhost:8001`

The web interface provides:
- Interactive map for location selection
- Time input controls (hour and minute selectors)
- Real-time probability updates
- Geographic variance visualization

### Console Interface
```bash
python predictor.py
```

The console interface supports:
- City name input (Houston, Miami, New Orleans, etc.)
- Coordinate input (latitude, longitude)
- Custom time selection
- Batch predictions

## API Reference

### Prediction Output
```python
{
    "precipitation_mm_hr": 0.1234,      # Predicted precipitation rate
    "probability": "36.94%",            # Formatted probability
    "probability_decimal": 0.3694,      # Decimal probability
    "category": "Light Rain",           # Intensity category
    "coordinates": {"lat": 29.76, "lon": -95.37},
    "time": "14:30"                     # Prediction time
}
```

### Supported Regions
- Latitude: 20°N to 35°N
- Longitude: 100°W to 75°W
- Optimized for Gulf Coast states (Texas, Louisiana, Mississippi, Alabama, Florida)

## Model Training

To retrain the model with new data:

```bash
python train_model.py
```

Ensure HDF5 files are placed in the `data/` directory following the GPM IMERG naming convention.

## File Structure

```
├── src/
│   ├── main.py              # Web application
│   └── logo/
│       └── nova_delta.jpeg
├── predictor.py             # Console interface
├── train_model.py           # Model training script
├── requirements.txt         # Dependencies
├── rain_prediction_model.joblib  # Trained model
└── README.md               # Documentation
```

## Dependencies

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning framework
- **Joblib**: Model serialization
- **NiceGUI**: Web framework
- **Folium**: Interactive mapping
- **H5PY**: HDF5 file handling
- **Pytz**: Timezone support

## Performance Notes

- Predictions are optimized for coastal and tropical regions
- Geographic variance algorithms ensure location-specific results
- Temporal encoding captures seasonal and diurnal patterns
- Probability calculations use sigmoid transformation for realistic ranges

## Limitations

- Model trained specifically for Gulf of Mexico region
- Requires minimum 0.01 mm/hr precipitation for meaningful probability calculation
- Performance may vary outside trained geographic bounds
- Real-time meteorological data not incorporated (uses statistical modeling)

## Contributing

When modifying the codebase:
1. Maintain feature consistency across interfaces
2. Preserve geographic variance calculations
3. Ensure probability calculations remain calibrated
4. Test predictions across multiple locations and times

## License

[License information]

## Contact

[Contact information]