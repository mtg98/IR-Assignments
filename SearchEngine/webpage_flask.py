from flask import Flask, render_template, request
from query_processing import query_pro
#Starting the Flask Server
app = Flask('server')

#Opening the search page
@app.route('/')
def home():
    return render_template('hello.html')

@app.route('/query',methods = ['POST', 'GET'])
#for finding the query 
def query():
    global query
    if request.method == 'POST':
        query=request.form['search']
        ranked_doc = query_pro(query)
        
        result = ranked_doc
        
        if result==0:
            return render_template("sorry.html",result=result)
        else:
            #opening the result page 
            return render_template("result.html",result = result)
    

if __name__=='__main__':
    #Running this code on localhost:5005
    app.run()