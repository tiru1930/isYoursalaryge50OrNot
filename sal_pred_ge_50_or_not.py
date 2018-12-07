import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

class SalaryGe50OrNot(object):
	"""docstring for SalaryGe50OrNot"""
	def __init__(self):
		super(SalaryGe50OrNot, self).__init__()
		pass 

	def prepareTrainData(self,data,isTrain=True):
		
		# print(data.head())
		data.rename(columns={'Employ No':'employ_no','education.num':'education_num',
					'marital.status':'marital_status','capital.gain':'capital_gain',
					'capital.loss':'capital_loss','hours.per.week':'hours_per_week',
					'native.country':'native_country'},inplace=True)
		# print(data.dtypes)
		# print(data.head())
		# print(data.isnull().values.sum()) #toget all null values rows count
		# print(data.isnull().sum())  #to get columnas which are having null value 

		#fill null values with mode
		data = data.fillna(data['workclass'].value_counts().index[0])
		data = data.fillna(data['occupation'].value_counts().index[0])
		data = data.fillna(data['native_country'].value_counts().index[0])

		# print(data.isnull().sum())  #to get columnas which are having null value 
		lb_make = LabelEncoder()
		for column in data.select_dtypes(include=['object']).columns:
			data[column] = lb_make.fit_transform(data[column])
		# print(data.head())
		# print(data.target.unique())

		if isTrain:
			# print(data[data.columns[1:]].corr()['target'][:])
			data.drop(["employ_no"],inplace=True,axis=1)
			msk = np.random.rand(len(data)) < 0.9
			self.train = data[msk]
			self.valid  = data[~msk]
			# print(self.train.dtypes)
		else:
			# print(data.dtypes)
			self.test = data 


	def trainmodel(self):

		train_y = np.array(self.train["target"].tolist())
		self.train.drop(["target"],inplace=True,axis=1)
		train_x = self.train.values
		col_tr  = self.train.columns

		test_y = np.array(self.valid["target"].tolist())
		self.valid.drop(["target"],inplace=True,axis=1)
		test_x = self.valid.values

		lrn = LogisticRegression(penalty = 'l1', C = .001, class_weight='balanced')
		lrn.fit(train_x, train_y)
		y_pred = lrn.predict(test_x)
		print ('Accuracy:', accuracy_score(test_y, y_pred))
		print ('F1 score:', f1_score(test_y,y_pred))

		#Below lines will tell correlation of the model with its coffesients 
		# coff=pd.DataFrame(lrn.coef_).T 
		# col=pd.DataFrame(col_tr).T 
		# print(coff)
		# print(col)
		self.model = lrn 

	def predictonNewdata(self):
		self.preds = self.model.predict(self.test.values)
		self.test["prediction"] = self.preds
	
		
def main():
	TRAIN_DATA = "train.csv"
	TEST_DATA =  "test.csv"
	NEW_TEST_DATA = "NEW_TEST_DATA__PREDICTED.CSV"
	data = pd.read_csv(TRAIN_DATA)
	SpG = SalaryGe50OrNot()
	SpG.prepareTrainData(data)
	SpG.trainmodel()
	Test = pd.read_csv(TEST_DATA)
	SpG.prepareTrainData(Test,isTrain=False)
	SpG.predictonNewdata()
	Test["predict"] = SpG.preds 
	Test.to_csv("test_with_predict.csv",index=False)
	new_test = pd.read_csv(NEW_TEST_DATA)
	SpG.prepareTrainData(new_test,isTrain=False)
	SpG.predictonNewdata()
	new_test["predict"] = SpG.preds 
	new_test.to_csv("new_test_with_predcit.csv",index=False)




if __name__ == '__main__':
	main()