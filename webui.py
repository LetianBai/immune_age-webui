from flask import Flask, render_template, request
import joblib
import numpy as np
import shap
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

def load_artifacts():
    model = joblib.load('model.pkl')
    scaler_final = joblib.load('scaler_final.pkl')
    background = np.load('background.npy')
    if os.path.exists('selected_features.csv'):
        selected_features = pd.read_csv('selected_features.csv', header=None).iloc[0].tolist()
    else:
        selected_features = ['Absolute Count of CD8+T cells', 'CD8+ Naive T Cells',
       'CD8+ Central Memory T Cells', 'CD4+ Activated Effector Memory T Cells',
       'CD8+CD122+', 'CX3CR1-CD27+', 'CD4+CD278+', 'CD8+CXCR5+',
       'Classical Monocytes', 'Monocyte-like Myeloid Cells']  # 请替换为实际特征名
    return model, scaler_final, background, selected_features

# 全局只初始化一次SHAP解释器
model, scaler_final, background, selected_features = load_artifacts()
explainer = shap.KernelExplainer(model.predict, background)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_age = None
    force_plot_html = ""
    user_values = {feat: request.form.get(feat, 0.0) for feat in selected_features}
    if request.method == 'POST':
        try:
            # 校验输入
            user_input = []
            for feat in selected_features:
                val = request.form.get(feat, None)
                if val is None or val == '':
                    raise ValueError(f"feature {feat} input is required")
                user_input.append(float(val))
            X_input = np.array(user_input).reshape(1, -1)
            X_input_scaled = scaler_final.transform(X_input)
            pred_age = model.predict(X_input_scaled)[0]
            shap_value = explainer.shap_values(X_input_scaled)
            force_plot = shap.force_plot(
                explainer.expected_value, shap_value[0], X_input_scaled[0],
                feature_names=selected_features, matplotlib=False
            )
            force_plot_html = shap.getjs() + force_plot.html()
        except Exception as e:
            force_plot_html = f"<div class='error'>Error: {e}</div>"
    return render_template(
        "webui.html",
        selected_features=selected_features,
        pred_age=pred_age,
        force_plot_html=force_plot_html,
        user_values=user_values
    )

if __name__ == '__main__':
    app.run(debug=True)
