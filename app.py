from flask import Flask, request, jsonify
from flask_cors import CORS   # للسماح بطلبات JS من GitHub Pages
from scipy import stats
import pandas as pd

app = Flask(__name__)
CORS(app)                     # نفتح CORS للنطاقات كلها (أو نحدد لاحقاً)

@app.route("/api/ttest", methods=["POST"])
def ttest():
    """
    JSON مُرسل من الصفحة:
    {
      "group1": [12, 15, 9, 14, 18],
      "group2": [10, 11, 8, 13, 12]
    }
    """
    try:
        data = request.get_json()
        g1 = data["group1"]
        g2 = data["group2"]
        t_stat, p_val = stats.ttest_ind(g1, g2)

        # نُرسل أيضاً معنوية الفرق ومتوسطين
        return jsonify({
            "t": round(float(t_stat), 4),
            "p": round(float(p_val), 4),
            "mean1": round(float(pd.Series(g1).mean()), 4),
            "mean2": round(float(pd.Series(g2).mean()), 4),
            "significant": p_val < 0.05
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# فحص سريع لباقي الاختبارات (لاحقاً نضيفها)
@app.route("/api/tests", methods=["GET"])
def list_tests():
    return jsonify(["ttest", "mannwhitney", "anova", "chi2", "corr", "regr"])

# لإطلاق الخدمة محلياً
if __name__ == "__main__":
    app.run(debug=True)
