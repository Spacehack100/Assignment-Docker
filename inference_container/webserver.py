from flask import Flask
from flask import render_template
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from joblib import load
app = Flask(__name__, template_folder="template")

@app.route('/')
def hello(): 
    return 'welcome to ads effectiveness prediction,<br>wich will predict if you buy a advertised product based on you gender, age and estimated yearly salary.<br>Enter the following URL: http://localhost:8080/predict/gender(0 for M,1 for F)/age/Estimated salary(per year)'

@app.route('/predict/<gender>/<age>/<estimatedSalary>')
def predict(gender,age,estimatedSalary):
    response = ""
    if gender == "0":
        response += "Gender = Male<br>"
    if gender == "1":
        response += "Gender = Female<br>"
    response += "Age =  %s<br>" % age
    response += "EstimatedSalary = %s<br>" % estimatedSalary
    
    clf = load('/ModelMap/model.joblib')
    result = clf.predict((np.array([gender,age,estimatedSalary])).reshape(1, -1))
    response += "The algorithm predicts that "
    if result == 0:
        response += "you will not buy the product."
    if result == 1:
        response += "you will buy the product."

    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)


