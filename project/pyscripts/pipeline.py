from __future__ import print_function
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover
from pyspark.mllib.clustering import LDA
from pyspark.sql import Row
from stop_words import get_stop_words
from time import localtime, strftime
import re
import nltk
from nltk.stem.porter import *



#####################################################################


DATAPATH = "/home/bng1290/project/data/"
LOGPATH = "/home/bng1290/project/logs/"
SMALLDATA = DATAPATH + "smallDump.txt"
TRAINING = DATAPATH + "training_set_tweets.txt"
TEST = DATAPATH + "test_set_tweets.txt"
FULLNELSON = False
TIME = strftime("%m%d%y_%H:%M:%S", localtime())
OUTPUT = open(LOGPATH + "output" + TIME + ".txt", "wr+")
RAWSEP = "#################RAWDATA###################\n"


#####################################################################


# Show the first record and number of records after pruning malformed lines
def data_peek(dataframe, output):
	print((RAWSEP + str(dataframe.first())), file=output)
	print("\nNUM_LINES: " + str(dataframe.count()), file=output)
	
def df2SVM(dataframe):
	return dataframe\
		.select(dataframe['C0'], dataframe['tfidf'])\
		.rdd.map(lambda row: [int(row.C0), row.tfidf]) 
	

# Initialize SparkContext and SQLContext
sc = SparkContext("local", "Pipeline")
sqlcontext = SQLContext(sc)

# Get english stop_words from NLTK
# en_stop = get_stop_words('en')

# Create tokenizer and write its output to 'words' column
tokenizer = Tokenizer(inputCol='cleantext', outputCol='words')

# Create stopper to remove stop words and output to stopped column
stopper = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='stopped')

# Create Porter Stemmer
stemmer = PorterStemmer()

# Create Term Frequency Hasher and write its output to features column
hashingTF = HashingTF(inputCol='stemmed', outputCol='features')

# Create TF-IDF Transformer and write its output to tfidf column
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol='tfidf')

# Add tokenizer and hashingTF to the Pipeline
# Fit/Transform the Pipeline on the DataFrame of raw data
# Fit the model to training data (or small data if not in full nelson mode)
# Then run the pipeline and predict for the test data (or small data again)
pipeline_stage1 = Pipeline(stages=[tokenizer, stopper])
pipeline_stage2 = Pipeline(stages=[hashingTF, idf])

if not FULLNELSON:
	# Read TSV data into a SparkSQL DataFrame
	raw_df = sqlcontext.read.format('com.databricks.spark.csv')\
		.options(header='false', delimiter='\t', mode='DROPMALFORMED',charset='ASCII')\
		.load(SMALLDATA)\
		.cache()
	data_peek(raw_df, OUTPUT)

	# Removes URL links or non-alphanumeric characters from a String
	def regex_replace(row):
		if row:
			return re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', row)
		else:
			return ""
	
	# Perform RegEx to remove URL links and any non-alphanumeric characters
	newColumnList = raw_df.columns
	newColumnList.insert(len(newColumnList), 'cleantext')
	clean_df = raw_df.map(lambda row: 
					(row + Row(regex_replace(row.C2)))).toDF(newColumnList)
	print(clean_df.take(20))
		
	# Run Stage1 of the pipline, Tokenization and Stop Word Removal	
	token_model = pipeline_stage1.fit(clean_df)
	tokenized = token_model.transform(clean_df) #need to tokenize and stop/stem etc
	print("\n" + str(tokenized.first()), file=OUTPUT)
	
	# Stems each String in a list of Strings
	def stem(tokens):
		if tokens:
			return [stemmer.stem(token) for token in tokens]
		else:
			return []
	
	# Append an RDD of stemmed token lists as a new column to the DataFrame
	newerColumnList = tokenized.columns
	newerColumnList.insert(len(newerColumnList), 'stemmed')
	stemmed_df = tokenized.map(lambda row: 
					(row + Row(stem(row.stopped)))).toDF(newerColumnList)	
	print("\n" + str(stemmed_df.take(3)), file=OUTPUT)
	
	# Run Stage 2 of the Pipline, perform TF hash and TF-IDF model
	model = pipeline_stage2.fit(stemmed_df)
	prediction = model.transform(stemmed_df)
	print("\n" + str(prediction.take(3)), file=OUTPUT)
	
	# Convert to LibSVM Form
	tf_rdd = df2SVM(prediction)
	print("\n" + str(tf_rdd.take(4)), file=OUTPUT)
	
	# Train LDA Model
	ldamodel = LDA.train(tf_rdd, k=5, seed=1)
	print("\n", file=OUTPUT)
	print(ldamodel.describeTopics(4), file=OUTPUT)
	print("\nVocab Size:", file=OUTPUT)
	print(ldamodel.vocabSize(), file=OUTPUT)
	
	
	
# else:
	# training_df = sqlcontext.read.format('com.databricks.spark.csv')\
		# .options(header-'false', delimiter='\t', mode='DROPMALFORMED')\
		# .load(TRAINING)\
		# .cache()
	# test_df = sqlcontext.read.format('com.databricks.spark.csv')\
		# .options(header-'false', delimiter='\t', mode='DROPMALFORMED')\
		# .load(TEST)\
		# .cache()
	# data_peek(training_df, OUTPUT)
	# data_peek(test_df, OUTPUT)
	# model = pipeline.fit(training_df)
	# prediction = model.transform(test_df)



















