from flask import Flask,render_template, request
from recommender import recommend_random, recommend_with_NMF, recommend_neighbourhood
from utils import movie_to_id, movies
app = Flask(__name__)

@app.route('/')
def hello():
    """
    Returns:
        hello is printed out
    """
    return render_template('index.html',name ='Prachi...!!',movies = movies.title.to_list())

@app.route('/recommend')
def recommendations():
    titles = movie_to_id(request.args.getlist('title'))
    ratings = list(map(int, request.args.getlist('Ratings')))
    query = dict(zip(titles,ratings))

    if request.args['algorithm']=='NMF':
        recs = recommend_with_NMF(query)
    elif request.args['algorithm']=='Random':
        recs = recommend_random()
    elif request.args['algorithm']=='K-Neighbours':
        recs = recommend_neighbourhood(query)
    else:
        return f"Function not defined"
    
    return render_template('recommend.html',recs =recs['title'])

if __name__=='__main__':
    app.run(debug=True,port=5000)