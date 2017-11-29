import numpy as np
import pandas as pd #used to import csv files easily
import lightgbm as lgb #gradient boosted tree builder
from sklearn.model_selection import train_test_split #used to easily split data into feature vectors and labels

data_folder = './data/'

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.2 ,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 100,
        'metric' : 'auc'
}

# First we want to load the data from csv files into pandas dataframes for easy handling
# We want pandas dataframes because we can manipulate them more easily than other types of objects
def load_data():
	print "Loading data..."
	train = pd.read_csv(data_folder + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
	test = pd.read_csv(data_folder + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})
	members = pd.read_csv(data_folder + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'},
                     parse_dates=['registration_init_time','expiration_date'])
	songs = pd.read_csv(data_folder + 'songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
	songs_extra = pd.read_csv(data_folder + 'song_extra_info.csv')
	print "Done."
	return train, test, members, songs, songs_extra


# Function to merge similar data fields together for convenience and fix some formatting
def merge_and_fix_data(train, test, songs, songs_extra, members):
	# All the information in songs can be concatenated to train and test based on song_id
	# Therefore, we use the pandas merge function to combine songs and train/test by song_id
	# The same idea applies to msno (membership number), and song_id
	print "Applying song merges..."
	train = train.merge(songs, how='left', on='song_id')
	test = test.merge(songs, how='left', on='song_id')
	train = train.merge(songs_extra, on = 'song_id', how = 'left')
	test = test.merge(songs_extra, on = 'song_id', how = 'left')

	songs_extra['song_year'] = songs_extra['isrc'].apply(convert_isrc_to_year)
	songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

	print "Done."

	# Members has the registration time and expiration year fields in single integer format with no separations 
	# Ex: November 25, 2017 is listed as 20171125, so we use datetime.year/month/day to separate these
	# We also calculate the number of days the user has been a member for later use.
	print "Applying fixes to members and merging..."

	members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

	members['registration_year'] = members['registration_init_time'].dt.year
	members['registration_month'] = members['registration_init_time'].dt.month
	members['registration_date'] = members['registration_init_time'].dt.day

	members['expiration_year'] = members['expiration_date'].dt.year
	members['expiration_month'] = members['expiration_date'].dt.month
	members['expiration_date'] = members['expiration_date'].dt.day

	members = members.drop(['registration_init_time'], axis=1) #we dont need these fields anymore
	members = members.drop(['expiration_date'], axis=1)

	train = train.merge(members, how='left', on='msno')
	test = test.merge(members, how='left', on='msno')
	print "Done."

	# Sometimes fields are left empty so we fill them with a placeholder to avoid errors
	print "Fixing empty fields..."
	train.song_length.fillna(200000,inplace=True)
	train.song_length = train.song_length.astype(np.uint32)
	train.song_id = train.song_id.astype('category')
	
	test.song_length.fillna(200000,inplace=True)
	test.song_length = test.song_length.astype(np.uint32)
	test.song_id = test.song_id.astype('category')

	print "Done."
	return train, test, members

# Helper function to convert isrc to a year
def convert_isrc_to_year(isrc):
	if type(isrc) == str:
		if int(isrc[5:7]) > 17:
			return 1900 + int(isrc[5:7])
		else:
			return 2000 + int(isrc[5:7])
	else:
		return np.nan


def train_and_validate(train, test):
	print "Preparing dev set..."
	for col in train.columns:
		if train[col].dtype == object:
			train[col] = train[col].astype('category')
			test[col] = test[col].astype('category')

	train_X = train.drop(['target'], axis=1)
	train_Y = y_train = train['target'].values

	# Split off part of the data to be used as dev set
	X_train, X_dev, Y_train, Y_dev = train_test_split(train_X, train_Y)

	lgb_train = lgb.Dataset(X_train, Y_train)
	lgb_dev = lgb.Dataset(X_dev, Y_dev)

	print "Done."

	# Train the model according to the parameters at the top of the file
	print "Training model..."
	lgb_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_dev, verbose_eval=5)
	print "Done."
	return lgb_model, test

def generate_predictions(lgb_model, test):
	print "Generating predictions for test set..."
	X_test = test.drop(['id'], axis=1)
	ids = test['id'].values
	predictions = lgb_model.predict(X_test)
	output = pd.DataFrame()
	output['ids'] = ids
	output['target'] = predictions
	output.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

if __name__ == "__main__":
	train, test, members, songs, songs_extra = load_data()
	train, test, members = merge_and_fix_data(train, test, songs, songs_extra, members)
	lgb_model, test = train_and_validate(train, test)
	generate_predictions(lgb_model, test)