from flask import Flask, render_template, request
import joblib
import numpy as np
import shap
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# 访问计数文件路径
VISITOR_COUNT_FILE = 'visitor_count.txt'

def load_artifacts():
    model = joblib.load('model.pkl')
    scaler_final = joblib.load('scaler_final.pkl')
    background = np.load('background.npy')
    if os.path.exists('selected_features.csv'):
        selected_features = pd.read_csv('selected_features.csv', header=None).iloc[0].tolist()
    else:
        selected_features = ['Absolute Count of CD8+T cells(/uL)', 'CD8+ Naive T Cells(%)',
       'CD8+ Central Memory T Cells(%)', 'CD4+ Activated Effector Memory T Cells(%)',
       'CD8+CD122+(%)', 'CX3CR1-CD27+(%)', 'CD4+CD278+(%)', 'CD8+CXCR5+(%)',
       'Classical Monocytes(%)', 'Monocyte-like Myeloid Cells(%)']  # 请替换为实际特征名
    return model, scaler_final, background, selected_features

def get_visitor_count():
    """获取访问人数"""
    if os.path.exists(VISITOR_COUNT_FILE):
        try:
            with open(VISITOR_COUNT_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def increment_visitor_count():
    """增加访问人数"""
    count = get_visitor_count()
    count += 1
    with open(VISITOR_COUNT_FILE, 'w') as f:
        f.write(str(count))
    return count

# 全局只初始化一次SHAP解释器
model, scaler_final, background, selected_features = load_artifacts()

# 在请求处理时动态初始化explainer（避免多进程问题）
explainer = None

def get_explainer():
    global explainer
    if explainer is None:
        explainer = shap.KernelExplainer(model.predict, background)
    return explainer

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_age = None
    force_plot_html = ""
    # 修改默认值处理方式，不使用默认值0.0
    user_values = {}
    for feat in selected_features:
        if request.method == 'POST':
            user_values[feat] = request.form.get(feat, '')
        else:
            user_values[feat] = ''
    
    # 增加访问计数
    visitor_count = increment_visitor_count()
    
    if request.method == 'POST':
        try:
            # 校验输入
            user_input = []
            missing_fields = []
            invalid_fields = []
            
            for idx, feat in enumerate(selected_features):
                val = request.form.get(feat, None)
                # 检查字段是否缺失
                if val is None or val == '':
                    missing_fields.append(feat)
                    continue
                
                # 检查字段是否为有效数字
                try:
                    num_val = float(val)
                    # 第一个字段（Absolute Count）无范围限制，其余百分比字段限制0-100
                    if idx > 0 and (num_val < 0 or num_val > 100):
                        invalid_fields.append(f"{feat} (value {num_val} is out of range 0-100)")
                    else:
                        user_input.append(num_val)
                except ValueError:
                    invalid_fields.append(feat)
            
            # 如果有缺失字段，抛出异常
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # 如果有无效字段，抛出异常
            if invalid_fields:
                raise ValueError(f"Invalid values for fields: {', '.join(invalid_fields)}")
                
            X_input = np.array(user_input).reshape(1, -1)
            X_input_scaled = scaler_final.transform(X_input)
            pred_age = model.predict(X_input_scaled)[0]
            
            # 获取explainer并计算SHAP值
            explainer_instance = get_explainer()
            shap_value = explainer_instance.shap_values(X_input_scaled)
            force_plot = shap.force_plot(
                explainer_instance.expected_value, shap_value[0], X_input_scaled[0],
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
        user_values=user_values,
        visitor_count=visitor_count
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
