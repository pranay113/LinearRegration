import joblib
salary = joblib.load("SalaryModel.pkl")
x=int(input("Enter your Experience")
print(salary.predict([[x]])
