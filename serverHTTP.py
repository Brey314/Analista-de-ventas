from flask import Flask, render_template

app=Flask(__name__)

@app.route('/inicio')
def inicio():
    return render_template('index.html')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/13')
def p13():
    return render_template('13.html')
@app.route('/14')
def p14():
    return render_template('14.html')
@app.route('/index1')
def index1():
    return render_template('index1.html')
@app.route('/login')
def login():
    return render_template('login.html')


app.run(host='0.0.0.0',port=80)

if __name__ == '__main__':
    app.run(debug=True)