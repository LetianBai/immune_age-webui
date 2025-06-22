# immune_age-webui
This repository provides a complete pipeline and web-based interface for predicting immune age based on peripheral blood lymphocyte subpopulations. The project includes data preprocessing, feature selection, model training, interpretation, and a user-friendly WebUI for clinical or research use.
## Features

- **Data Preprocessing:** Handles missing values, encodes categorical variables, and standardizes features.
- **Feature Selection:** Uses Lasso regression for automatic feature selection.
- **Model Training:** Supports multiple regression models with hyperparameter tuning; Gradient Boosting Regressor is used as the final model.
- **Model Interpretation:** Integrates SHAP for feature importance and individual prediction explanations.
- **WebUI:** Flask-based interface for easy input, prediction, and visualization of results.

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/LetianBai/immune_age-webui/.git
   cd immune-age-webui
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare model artifacts:**
   - Place `model.pkl`, `scaler_final.pkl`, `background.npy`, and (optionally) `selected_features.csv` in the project root.
   - If you wish to retrain the model, refer to the provided Jupyter notebook for the full pipeline.

4. **Run the WebUI:**
   ```
   python webui.py
   ```
   The interface will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

- Open the WebUI in your browser.
- Enter the required lymphocyte subpopulation values.
- Click "Predict" to obtain the immune age and view the SHAP-based feature contribution plot.

## Example

*Figure 1: Example screenshot of the immune age prediction WebUI interface.*

![WebUI Example](docs/webui_example.png)  
*(Please replace with your actual screenshot.)*

## File Structure

- `webui.py` — Main Flask application.
- `templates/webui.html` — HTML template for the interface.
- `model.pkl`, `scaler_final.pkl`, `background.npy` — Model and preprocessing artifacts.
- `selected_features.csv` — (Optional) List of selected features.
- `requirements.txt` — Python dependencies.
- `age.ipynb` — Jupyter notebook for data analysis and model training.


## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, please open an issue or contact [ryancrz@qq.com].
