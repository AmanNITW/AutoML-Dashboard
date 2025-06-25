from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
import uuid
import shutil
import pickle
from pycaret.classification import setup as setup_clf, compare_models as compare_clf, pull as pull_clf, plot_model as plot_clf, save_model as save_clf, load_model as load_clf, predict_model as predict_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg, plot_model as plot_reg, save_model as save_reg, load_model as load_reg, predict_model as predict_reg

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df_global = None
target_column = None
best_model_path = None
problem_type = None
feature_columns = []

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    global df_global
    df_global = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
    session['filename'] = filename
    return redirect(url_for('select_target'))

@app.route('/select_target')
def select_target():
    global df_global
    if df_global is None:
        return redirect(url_for('index'))

    valid_columns = [col for col in df_global.columns if df_global[col].nunique() > 1 and df_global[col].nunique() < 50]
    return render_template('select_target.html', columns=valid_columns)

@app.route('/download_model')
def download_model():
    global best_model_path
    if best_model_path and os.path.exists(best_model_path):
        return send_file(best_model_path, as_attachment=True)
    return "Model not available. Run AutoML first."

@app.route('/analyze', methods=['POST'])
def analyze():
    global df_global, target_column, best_model_path, problem_type, feature_columns
    target_column = request.form['target']

    if df_global is None or target_column not in df_global.columns:
        return "Invalid target or dataset not loaded", 400

    try:
        df_filtered = df_global[df_global[target_column].map(df_global[target_column].value_counts()) > 1]

        shutil.rmtree(PLOT_FOLDER)
        os.makedirs(PLOT_FOLDER, exist_ok=True)

        n_unique = df_filtered[target_column].nunique()
        dtype = df_filtered[target_column].dtype
        plot_paths = []

        if dtype == 'object' or n_unique <= 10:
            setup_clf(data=df_filtered, target=target_column, verbose=False, session_id=123, n_jobs=1)
            best_model = compare_clf()
            results_df = pull_clf()
            problem_type = "Classification"
            feature_columns = [col for col in df_filtered.columns if col != target_column]

            model_name = f"best_model_clf_{uuid.uuid4().hex}"
            save_clf(best_model, model_name=os.path.join(MODEL_FOLDER, model_name))
            best_model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")

            plots = ['confusion_matrix', 'auc', 'feature', 'pr', 'learning']
            for p in plots:
                try:
                    plot_clf(best_model, plot=p, save=True)
                    src = f"{p}.png"
                    dst = os.path.join(PLOT_FOLDER, src)
                    os.rename(src, dst)
                    plot_paths.append((p.upper(), f"plots/{src}"))
                except Exception as e:
                    print(f"Error plotting {p}: {e}")

        elif dtype in ['int64', 'float64'] and n_unique > 10:
            setup_reg(data=df_filtered, target=target_column, verbose=False, session_id=123, n_jobs=1)
            best_model = compare_reg()
            results_df = pull_reg()
            problem_type = "Regression"
            feature_columns = [col for col in df_filtered.columns if col != target_column]

            model_name = f"best_model_reg_{uuid.uuid4().hex}"
            save_reg(best_model, model_name=os.path.join(MODEL_FOLDER, model_name))
            best_model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")

            plots = ['residuals', 'error', 'feature', 'cooks', 'learning']
            for p in plots:
                try:
                    plot_reg(best_model, plot=p, save=True)
                    src = f"{p}.png"
                    dst = os.path.join(PLOT_FOLDER, src)
                    os.rename(src, dst)
                    plot_paths.append((p.upper(), f"plots/{src}"))
                except Exception as e:
                    print(f"Error plotting {p}: {e}")
        else:
            return "Target column is not suitable for modeling."

        # EDA Plots
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(PLOT_FOLDER, 'heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        pairplot_path = os.path.join(PLOT_FOLDER, 'pairplot.png')
        try:
            sns.pairplot(df_filtered.select_dtypes(include=['float64', 'int64']).sample(n=min(100, len(df_filtered))))
            plt.savefig(pairplot_path)
            plt.close()
        except Exception as e:
            print(f"Pairplot failed: {e}")
            pairplot_path = None

        boxplot_paths = []
        numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col != target_column and df_filtered[col].nunique() > 5:
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=df_filtered, y=col)
                plt.title(f"Boxplot for {col}")
                fname = f"boxplot_{col}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_FOLDER, fname))
                plt.close()
                boxplot_paths.append((f"Boxplot for {col}", f"plots/{fname}"))

        plt.figure(figsize=(6, 4))
        sns.histplot(df_filtered[target_column], kde=True)
        plt.title(f"Distribution of Target: {target_column}")
        target_dist_path = os.path.join(PLOT_FOLDER, 'target_dist.png')
        plt.tight_layout()
        plt.savefig(target_dist_path)
        plt.close()

        describe_table = df_filtered.describe().to_html(classes="table table-striped")

        return render_template('result.html',
                               title=f"AutoML {problem_type} Results for {target_column}",
                               table=results_df.to_html(classes="table table-bordered", index=False),
                               describe_table=describe_table,
                               plot_paths=plot_paths,
                               boxplot_paths=boxplot_paths,
                               heatmap_path='plots/heatmap.png',
                               target_dist_path='plots/target_dist.png',
                               pairplot_path='plots/pairplot.png' if pairplot_path else None)

    except Exception as e:
        return f"<h3>Error during AutoML: {str(e)}</h3><a href='/'>Go Back</a>"

@app.route('/predict_new_data', methods=['GET', 'POST'])
def predict_new_data():
    global best_model_path, problem_type
    if request.method == 'GET':
        return render_template('predict.html')

    if best_model_path is None or not os.path.exists(best_model_path):
        return "Model not found. Please run AutoML first."

    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    new_df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

    try:
        model_name = os.path.splitext(os.path.basename(best_model_path))[0]
        model_path = os.path.join(MODEL_FOLDER, model_name)

        if problem_type == "Classification":
            model = load_clf(model_path)
            predictions = predict_clf(model, data=new_df)
        elif problem_type == "Regression":
            model = load_reg(model_path)
            predictions = predict_reg(model, data=new_df)
        else:
            return "Model type unknown"

        pred_html = predictions.to_html(classes="table table-striped", index=False)
        return render_template('prediction_result.html', table=pred_html)

    except Exception as e:
        return f"<h3>Error during Prediction: {str(e)}</h3><a href='/'>Go Back</a>"

@app.route('/predict_single_entry', methods=['GET', 'POST'])
def predict_single_entry():
    global best_model_path, feature_columns, problem_type
    if best_model_path is None:
        return "Model not trained yet."

    if request.method == 'POST':
        user_input = {col: request.form[col] for col in feature_columns}
        input_df = pd.DataFrame([user_input])

        try:
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='ignore')

            model_name = os.path.splitext(os.path.basename(best_model_path))[0]
            model_path = os.path.join(MODEL_FOLDER, model_name)

            if problem_type == "Classification":
                model = load_clf(model_path)
                prediction = predict_clf(model, data=input_df)
            else:
                model = load_reg(model_path)
                prediction = predict_reg(model, data=input_df)

            result_html = prediction.to_html(classes="table table-bordered", index=False)
            return render_template("prediction_result.html", table=result_html)
        except Exception as e:
            return f"<h3>Error in prediction: {e}</h3>"

    return render_template("predict_single_entry.html", columns=feature_columns)

if __name__ == '__main__':
    app.run(debug=True)
