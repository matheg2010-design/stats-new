import os, json, math, itertools
import pandas as pd
import numpy as np
from scipy import stats
from flask import Flask, request, jsonify
from flask_cors import CORS

PORT = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
CORS(app)

# ---------- 1. T-TEST ----------
@app.route("/api/ttest", methods=["POST"])
def ttest():
    data = request.get_json()
    g1, g2 = data["group1"], data["group2"]
    t, p = stats.ttest_ind(g1, g2, equal_var=True)
    return jsonify({
        "test": "Independent t-test",
        "t": t, "p": p,
        "mean1": np.mean(g1), "mean2": np.mean(g2),
        "sig": p < 0.05
    })

# ---------- 2. MANN-WHITNEY (non-parametric) ----------
@app.route("/api/mannwhitney", methods=["POST"])
def mannwhitney():
    data = request.get_json()
    g1, g2 = data["group1"], data["group2"]
    u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return jsonify({
        "test": "Mann-Whitney U",
        "U": u, "p": p,
        "median1": np.median(g1), "median2": np.median(g2),
        "sig": p < 0.05
    })

# ---------- 3. ONE-WAY ANOVA ----------
@app.route("/api/anova", methods=["POST"])
def anova():
    data = request.get_json()          # {"groups": [[...],[...],[...]]}
    groups = data["groups"]
    f, p = stats.f_oneway(*groups)
    return jsonify({
        "test": "One-Way ANOVA",
        "F": f, "p": p,
        "means": [np.mean(g) for g in groups],
        "sig": p < 0.05
    })

# ---------- 4. CHI-SQUARE ----------
@app.route("/api/chi2", methods=["POST"])
def chi2():
    data = request.get_json()          # {"observed": [[...],[...]]}
    obs = np.array(data["observed"])
    chi2, p, dof, expected = stats.chi2_contingency(obs)
    return jsonify({
        "test": "Chi-Square",
        "chi2": chi2, "p": p, "dof": dof,
        "expected": expected.tolist(),
        "sig": p < 0.05
    })

# ---------- 5. PEARSON CORRELATION ----------
@app.route("/api/corr", methods=["POST"])
def corr():
    data = request.get_json()          # {"x": [...], "y": [...]}
    x, y = data["x"], data["y"]
    r, p = stats.pearsonr(x, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return jsonify({
        "test": "Pearson correlation",
        "r": r, "p": p,
        "regression": {"slope": slope, "intercept": intercept},
        "sig": p < 0.05
    })

# ---------- 6. LINEAR REGRESSION (متعدد) ----------
@app.route("/api/regression", methods=["POST"])
def regression():
    data = request.get_json()          # {"y": [...], "x": [[...],[...]]}  x مصفوفة 2D
    y = np.array(data["y"])
    x = np.array(data["x"]).T          # كل عمود متغير مستقل
    results = stats.linregress(*x.T) if x.shape[1] == 1 else \
              stats.multivariate_regression(x, y)
    # نرجع أول نموذج بسيط
    slope, intercept, r_val, p_val, std_err = stats.linregress(x[:,0], y)
    return jsonify({
        "test": "Simple Linear Regression",
        "slope": slope, "intercept": intercept,
        "r": r_val, "p": p_val,
        "sig": p_val < 0.05
    })

# ---------- 7. قائمة الاختبارات ----------
@app.route("/api/tests", methods=["GET"])
def tests_list():
    return jsonify(["ttest", "mannwhitney", "anova", "chi2", "corr", "regression"])

# ---------- تشغيل ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
