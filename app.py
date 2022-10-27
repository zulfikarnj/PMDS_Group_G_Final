import re
from unittest import result
from flask import Flask, render_template, request
import joblib
import pandas as pd
from pyparsing import replaceWith
from src.preprocess import decode_input, map_year, map_location, categorize_input

model = joblib.load("output/model_used.pickle")
skala = joblib.load("output/standard_scaler.pkl")

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data = {}
    data['Location'] = [str(request.form['Location'])]
    data['WeekCase'] = [int(request.form['WeekCase'])]
    data['Year'] = [int(request.form['Year'])]
    data['WeekCasePerMil'] = [int(request.form['WeekCasePerMil'])]
    data['WeekDeath'] = [int(request.form['WeekDeath'])]
    data['WeekDeathPerMil'] = [int(request.form['WeekDeathPerMil'])]
    data['TotalVac'] = [int(request.form['TotalVac'])]
    data['PeopleVac'] = [int(request.form['PeopleVac'])]
    data['PeopleFullVac'] = [int(request.form['PeopleFullVac'])]
    data['TotalBoost'] = [int(request.form['TotalBoost'])]
    data['DailyVac'] = [int(request.form['DailyVac'])]
    data['TotalVacPerHun'] = [int(request.form['TotalVacPerHun'])]
    data['PeopelVacPerHun'] = [int(request.form['PeopleVacPerHun'])]
    data['PeopleFullVacPerHun'] = [int(request.form['PeopleFullVacPerHun'])]
    data['TotalBoostPerHun'] = [int(request.form['TotalBoostPerHun'])]
    data['DailyVacPerHun'] = [int(request.form['DailyVacPerHun'])]
    data['DailyPeopleVac'] = [int(request.form['DailyPeopleVac'])]
    data['DailyPeopleVacPerHun'] = [int(request.form['DailyPeopleVacPerHun'])]

    df = pd.DataFrame.from_dict(data)
    df_transform = decode_input(df, save_file=False)
    df_transform['Location'] = map_location(df_transform['Location'])
    df_transform['Year'] = df_transform['Year'].apply(map_year)
    
    num_col, cat_col = categorize_input(df_transform)
    
    df_transform[num_col] = skala.transform(df_transform[num_col])
    
    pred = int(model.predict(df_transform))
    return render_template('after.html', result = pred)
    
    

if __name__ == "__main__":
    app.run(debug=True)
