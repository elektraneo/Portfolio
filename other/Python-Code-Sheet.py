PYTHON CODES

##Python tips
##Print all output not just last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

##To Include Markdown in Your Code’s Output (Colors, Bold, etc.)
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

printmd("**bold and blue**", color="blue")

##Environments/modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
plt.style.use('ggplot')

##Specific things
from sklearn import naive_bayes
from numpy import *
from string import ascii_letters

##CONDA
  ##CALL CONDA IN TERMINAL 'conda' to install
	##install specific package
	conda install -c glemaitre imbalanced-learn
	##in worksheet
	from imblearn.over_sampling import SMOTE

##Timer
from timeit import default_timer
st = default_timer()
runtime = default_timer() - st
print ("Elapsed time(sec): ", round(runtime,2))

##Reading importing files into dataframes
##https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html

df = pd.read_table('german.data.txt', delim_whitespace=True, 
	names=["Status","Month"])

movies_cols = ['movieId', 'title', 'genres']
movies = pd.read_table('movies.dat', names=movies_cols,
                       index_col=False,
                       delimiter='::',
                       error_bad_lines=False,
                       encoding = 'ISO-8859-1')

df = pd.DataFrame(data, columns=['Age', 'Attrition', 
	'BusinessTravel', 'DailyRate'])

##print string
print("Number of mislabeled points out of a total %d points : %d")
		##to edit the string https://docs.python.org/3.1/library/string.html
		from string import ascii_letters

##View data

df.head(5) ##first 5 lines of daatframe
df.shape[:] ##what is the dimensions?
	trn_x.shape, trn_y.shape

##Sort data
movies.sort_values(by='movieId', inplace=True)
movies.reset_index(inplace=True, drop=True)

##1. Pre-processing
	##Drop columns
	df.drop(['DailyRate', 'EmployeeCount',])

	##Column names
	columns.values

	##Binarize the data
	df['Gender'].replace(['Female','Male'],[0,1],inplace=True)
	df['Attrition'].replace(['No','Yes'],[0,1],inplace=True)

	##move columns to a separate dataframe
	todummy_list = ['BusinessTravel']

		##combine dataframes
		x = dummy_df(df, todummy_list)

	##Split columns into 2
	movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
	movies.year = pd.to_datetime(movies.year, format='%Y')
	movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
	movies.title = movies.title.str[:-7]

	#Split columns out to true and false
	genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
	genres_unique = pd.DataFrame(genres_unique, columns=['genre']) # Format into DataFrame to store later
	movies = movies.join(movies.genres.str.get_dummies().astype(bool))
	movies.drop('genres', inplace=True, axis=1)

	##data split, training and test sets
	##create x and y, where x is the data set with predicted value. y is only the prediction value
	x= df.drop(['Attrition'], axis = 1)
	y = df.Attrition

	from sklearn.model_selection import train_test_split
			X_train, X_test, y_train, y_test = train_test_split(
     		data, target, test_size=0.3, random_state=100)

	##Check for nulls
	print(df.isnull().sum())
	newclaim = claim.dropna()

	##Investigate nulls
	nans = lambda claim: claim[claim.isnull().any(axis=1)]
	nans(claim)

	##fill nulls with random
	claim["claim_date"].fillna(lambda x: random.choice(claim[claim[claim_date] != np.nan]["claim_date"]), inplace =True)

	##Replace 0 with NaN
	df.replace(0, np.NaN)

	##Replace strings or words partially delete
	claim["claim_date"] = claim["claim_date"].str.replace("/","").astype(float)

	##unique variables
	claim.claim_reason.unique()

	##unique counts
	claim.claim_id.nunique(dropna=True)

	##value counts
	claim.claim_reason.value_counts()

	##label encoder
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	claim["claim_reason_e"] = le.fit_transform(claim["claim_reason"])

	##data types
	claim.dtypes

	##change data type
	claim.claim_date = claim.claim_date.astype(np.int64)

	##round rounding
	claim.amount.round(decimals=0)

	##export to csv
	claim.to_csv('claim_cleaned.csv',sep=',',header=True,index=False)

	##binning
	bins = [0,0.000799, 0.002007, 0.010719, 1]
	labels = [0,1,2,3]
	final['binned'] = pd.cut(final['amount'], bins=bins, labels=labels)
	print (final[['amount','binned']])

	##data matrix
	finalmatrix = final.as_matrix(columns=None)

## 1.5 Explore
##https://seaborn.pydata.org/examples/index.html
	##Histogram
	%matplotlib inline
	def plot_histogram(x):
	    plt.hist(x, color='gray', alpha=0.5)
	    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
	    plt.xlabel("Value")
	    plt.ylabel("Frequency")
	    plt.show()
	    plot_histogram(x['Attrition'])

	##Boxplot
    	df.boxplot(column='MonthlyIncome', by='Attrition')

    ##Correlation
    from scipy.stats.stats import pearsonr ##quantifies the degree to which a relationship between two variables can be described by a line.
		pearsonr (df.Age, df.MonthlyIncome)

		##Correlation diagram with a heatmap ontop
		corr = df.corr(method= 'pearson') ##define pearson
		mask = np.zeros_like(corr, dtype=np.bool) ##zeros_like returns an array of zeros with same shape as given array
		mask[np.triu_indices_from(mask)] = True ##Return the indices for the upper-triangle of arr.

		sns.set(style="white")
		cmap = sns.diverging_palette(100, 10, as_cmap=True) ##create the colour scheme for heat map
		sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
		           square=True, linewidths=.5, cbar_kws={"shrink": .5}) ##define it

	##Coreelation and p-value
	from scipy.stats.stats import spearmanr ##spearman is not restricted to linear relationship but uses
											##monotonic association (only strictly increasing or decreasing, but not mixed) between two variables and relies on the rank order of values.
		spearmanr (df.JobSatisfaction, df.JobInvolvement)

	##Scatterplot
		line = plt.figure()
		x = df.Age
		y = df.MonthlyIncome
		plt.title("Scatterplot of {var_name} and {var_name2}".format(var_name=x.name, var_name2=y.name))
		plt.xlabel("Age")
		plt.ylabel("MonthlyIncome")
		plt.plot(x,y,"o")

	##Summary Statistics Stats
	amount1.describe().apply(lambda x: format(x, 'f'))

	##Median of each column in dataframe
		df.median(axis=0)

	##Minimum and maxiumum of each column in dataframe
		df.min(axis=0)
		df.max(axis=0)

	##Standard Deviation
	claim.amount.std(axis=0)

	##Count how many of each variable in each column
	df.Predict.value_counts() ##predict is the column name

	##mean average
	claim.amount.mean(axis=0)

	##Outliers
	def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.96*iqr
    ceiling = q3 + 1.96*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values
    tukey_indices, tukey_values = find_outliers_tukey(claim['amount'])
	print(np.sort(tukey_values))

	##Outlier Box Plot
	i = 'amount'
	sns.set_style("whitegrid")
	plt.subplot(212)
	plt.xlim(x[i].min(), x[i].max()*1.1)
	sns.boxplot(x=x[i])
	sns.set(rc={'figure.figsize':(14.7,9.27)})

	##Find amount from outlier
	claim.loc[claim['amount'] == 6331949]

## 2. Algorithm

	##GaussianNB
		from sklearn.naive_bayes import GaussianNB
			gnb = GaussianNB()

	##Neural Network
		from sklearn.neural_network import MLPClassifier
		mlpc = MLPClassifier() ##what is the classifier dataframe?
		mlpc.fit(trn_x, trn_y) ##fit the data frames
		tst_preds = mlpc.predict(tst_x) ##whats the prediction accuracy?

		##Enforce number of hidden layers
		hn = 100
		mlpc = MLPClassifier(hidden_layer_sizes=(hn))
		mlpc.fit(trn_x, trn_y)
		tst_preds1 = mlpc.predict(tst_x)  

		performance = np.sum(tst_preds == tst_y) / float(tst_preds.size)
		print ("#.Hidden Neurons", hn)
		print ("Accuracy", performance)

		##with a for loop
		hn = 100
		mlpc_ = MLPClassifier(hidden_layer_sizes=(hn,))
		perf_records_ = []
		for i in range(10):
		    trn_x, tst_x, trn_y, tst_y = train_test_split(x, y)    
		    mlpc.fit(trn_x, trn_y)
		    tst_preds = mlpc.predict(tst_x)
		    performance = np.sum(tst_preds == tst_y) / float(tst_preds.size)
		    perf_records_.append(performance)

		print ("#.Hidden Neurons", hn)
		print ("Accuracy", perf_records_)
		print ("Avg. Accuracy", np.mean(perf_records_))

		##Set one parameter, hidden layer to MULTIPLE values and 
		##compute the AVERAGE performance
		hidden_neuron_nums = list(range(2,10)) + list(range(10,100,10)) + list(range(100,200,25))
		total_performance_records = []
		for hn in hidden_neuron_nums:
		    mlpc_ = MLPClassifier(hidden_layer_sizes=(hn,))
		    perf_records_ = []
		    for i in range(10):
		        trn_x, tst_x, trn_y, tst_y = train_test_split(x, y)    
		        mlpc_.fit(trn_x, trn_y)
		        tst_p_ = mlpc_.predict(tst_x)
		        performance = np.sum(tst_p_ == tst_y) / float(tst_preds.size)
		        perf_records_.append(performance)
		    total_performance_records.append(np.mean(perf_records_))
		    print ("Evaluate hidden layer {} done, accuracy {:.2f}".format(
		        hn, total_performance_records[-1]))

##3. Evaluate

from sklearn.metrics import confusion_matrix
	confusion_matrix(target, y_pred)

from sklearn.metrics import accuracy_score
	accuracy_score(target, y_pred)

from sklearn.metrics import classification_report
	target_names = ['Attrition 0', 'Attrition 1']
	print(classification_report(target, y_pred, target_names=target_names))

from sklearn.metrics import roc_auc_score
	roc_auc_score(target, predicted)

from sklearn.metrics import precision_score
	pres = precision_score(target, predicted)

from sklearn.metrics import precision_recall_curve
	precision, recall, _ = precision_recall_curve(target, predicted)
	plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
	          pres))





