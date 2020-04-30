from flask import Flask, render_template,request,redirect, url_for, session
from flask_nav import Nav 
from flask_nav.elements import Navbar, View
from flask_bootstrap import Bootstrap
from skripsi_classes import *
import pickle

nav = Nav()
app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = "dfsgdhfusdshjfdshjudsj7348236"

@nav.navigation()
def mynavbar():
    return Navbar(
        'Youtube Cyberbullying Comment Filter',
        View('Home', 'index'),
        View('Hasil', 'result'),
        View('Tentang', 'about'),
    )

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/result-table")
def result_table():
	hasil_prediksi = pickle.load( open( "hasil_prediksi.p", "rb" ) )
	komentar_prediksi = pickle.load( open( "daftar_komentar.p", "rb" ) )
	return render_template("result_table.html", hasil_prediksi = hasil_prediksi, komentar_prediksi = komentar_prediksi)

@app.route("/result")
def result():
	videoid = session['videoid']
	kernel = session['kernel']
	nilai_c = session['nilai_c']
	degree = session['degree']
	coef0 = session['coef0']

	comments = youtube_mining.comment_mining(videoid,100,"")
	video_information = youtube_mining.video_description(videoid)

	stemmer = preprocessing.stemming_create()
	sentences = []
	classes = []
	sentences_test = []
	with open('processed.csv', 'r') as f:
		object1 = csv.reader(f);
		list1 = list(object1)

	file1 = open("kbbi.txt","r")
	dictionary = set(word_tokenize(file1.read()))
	file1.close()

	[sentences.append(list1[x][0]) for x, text in enumerate(list1)]
	[classes.append(list1[x][1]) for x, text in enumerate(list1)]

	[sentences_test.append(comments[x]['content']) for x, text in enumerate(comments)]
	for x in range(len(sentences_test)):
		sentences_test[x] = preprocessing.remove_punctuation(sentences_test[x])
		sentences_test[x] = sentences_test[x].casefold()
		sentences_test[x] = stemmer.stem(sentences_test[x])
		sentences_test[x] = preprocessing.stopword_removal(word_tokenize(sentences_test[x])) 
		sentences_test[x] = preprocessing.normalization(sentences_test[x],dictionary)

	vectorizer = TfidfVectorizer()
	Train_X_Tfidf = vectorizer.fit_transform(sentences)
	Tfidf_Komentar = vectorizer.transform(sentences_test)
	classify = classification.svm(Train_X_Tfidf,classes,Tfidf_Komentar ,kernel, int(degree), float(nilai_c), float(coef0))
	pickle.dump( classify, open( "hasil_prediksi.p", "wb" ) )
	pickle.dump( comments, open( "daftar_komentar.p", "wb" ) )
	print(classify)
	for x in range(len(classify)):
		print(classify[x])
		if(str(classify[x]) == "1"):
			del comments[x]
	print(len(classify))
	return render_template('result.html', request1 = comments, request2 = video_information, linkvideo = "https://www.youtube.com/watch?v=" + videoid)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/", methods = ['GET', 'POST'])
def formtest():
	if request.method == 'POST':
		session['videoid'] = request.form['ylink']
		session['kernel'] = request.form['kernel']
		session['nilai_c'] = request.form['nilai_c']
		session['degree'] = request.form['degree']
		session['coef0'] = request.form['coef0']
		return redirect(url_for('result'))
nav.init_app(app)

if __name__ == "__main__":
    app.run(debug=True)
