import numpy as np
import pandas as pd #used to import csv files easily
import lightgbm as lgb #gradient boosted tree builder
from sklearn.model_selection import train_test_split #used to easily split data into feature vectors and labels
from sklearn.metrics import mean_squared_error
import time
data_folder = '/Users/DJ-M/Desktop/Univ/OSU/Fall17/CS534/Project/data/'

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.2 ,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 1.0,
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

# All feature engineering functions
def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

def song_length_categorise (length):
	if 0<length < 1000:
		return 'short'
	elif 1000<=length <2000 :
		return 'medium'
	elif length >= 2000:
		return 'long'
	else:
		return 'none'

def membership_days_categorise (length):
	if 0<length < 500:
		return 'short'
	elif 500<=length <1500 :
		return 'medium'
	elif length >= 1500:
		return 'long'
	else:
		return 'none'
	
def add_new_features(train, test):
	def smaller_song(x):
		if x < avg_song_length:
			return 1
		return 0

	def count_song_played(x):
		try:
			return _dict_count_song_played_train[x]
		except KeyError:
			try:
				return _dict_count_song_played_test[x]
			except KeyError:
				return 0

	def count_artist_played(x):
		try:
			return _dict_count_artist_played_train[x]
		except KeyError:
			try:
				return _dict_count_artist_played_test[x]
			except KeyError:
				return 0

	print "Adding new features:"
	print "Counting number of genre_ids..."
	train['genre_ids'] = train['genre_ids'].cat.add_categories(['no_genre_id'])
	test['genre_ids'] = test['genre_ids'].cat.add_categories(['no_genre_id'])
	train['genre_ids'].fillna('no_genre_id',inplace=True)
	test['genre_ids'].fillna('no_genre_id',inplace=True)
	train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
	test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)
	
	print "Counting number of lyricist..."
	train['lyricist'] = train['lyricist'].cat.add_categories(['no_lyricist'])
	test['lyricist'] = test['lyricist'].cat.add_categories(['no_lyricist'])
	train['lyricist'].fillna('no_lyricist',inplace=True)
	test['lyricist'].fillna('no_lyricist',inplace=True)
	train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
	test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)
	
	print "Counting number of composers..."
	train['composer'] = train['composer'].cat.add_categories(['no_composer'])
	test['composer'] = test['composer'].cat.add_categories(['no_composer'])
	train['composer'].fillna('no_composer',inplace=True)
	test['composer'].fillna('no_composer',inplace=True)
	train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
	test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)
	
	print "Counting number of artists..."
	train['artist_name'] = train['artist_name'].cat.add_categories(['no_artist'])
	test['artist_name'] = test['artist_name'].cat.add_categories(['no_artist'])
	train['artist_name'].fillna('no_artist',inplace=True)
	test['artist_name'].fillna('no_artist',inplace=True)
	train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
	test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)
	
	print "Comparing composers, artists and lyricists..."
	train['composer=artist'] = (np.asarray(train['composer']) == np.asarray(train['artist_name'])).astype(np.int8)
	test['composer=artist'] = (np.asarray(test['composer']) == np.asarray(test['artist_name'])).astype(np.int8)
	train['artist=lyricist'] = (np.asarray(train['artist_name']) == np.asarray(train['lyricist'])).astype(np.int8)
	test['artist=lyricist'] = (np.asarray(test['artist_name']) == np.asarray(test['lyricist'])).astype(np.int8)
	train['composer=lyricist'] = (np.asarray(train['composer']) == np.asarray(train['lyricist'])).astype(np.int8)
	test['composer=lyricist'] = (np.asarray(test['composer']) == np.asarray(test['lyricist'])).astype(np.int8)
	train['composer=lyricist=artist'] = (np.asarray(train['composer=artist']) == np.asarray(train['artist=lyricist'])).astype(np.int8)
	test['composer=lyricist=artist'] = (np.asarray(test['composer=artist']) == np.asarray(test['artist=lyricist'])).astype(np.int8)

	print "Converting languages to boolean..."
	train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
	test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)
	
	print "Categorizing song_length..."
	avg_song_length = np.mean(train['song_length'])
	
	_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
	_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}

	train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
	test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)
	
	_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
	_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}

	train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
	test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)

	train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
	test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

	        #categorising song_length
	print "Categorizing membership_days..."
	train['song_length_cat'] = train['song_length'].apply(song_length_categorise).astype('category')
	test['song_length_cat'] = test['song_length'].apply(song_length_categorise).astype('category')

	
        #categorising membership_days
	train['membership_days_cat'] = train['membership_days'].apply(song_length_categorise).astype('category')
	test['membership_days_cat'] = test['membership_days'].apply(song_length_categorise).astype('category')
	
	print "Done."
	return train, test
	
	
def one_hot_enc(train):
	init_headers = []
	one_hot_sst = pd.get_dummies(train.source_system_tab)
	one_hot_ssn = pd.get_dummies(train.source_screen_name)
	one_hot_st = pd.get_dummies(train.source_type)
	init_headers += list(one_hot_sst) + list(one_hot_ssn) + list(one_hot_st)
	train.drop(['source_system_tab','source_screen_name','source_type'], axis=1, inplace=True)
	train = pd.concat([train, one_hot_sst, one_hot_ssn, one_hot_st], axis=1)
	for head in init_headers:
	   train[head] = train[head].astype(np.uint8)
	   
	one_hot_final = train
	to_one = ['gender', 'registered_via', 'registration_month', 'registration_date', 'expiration_year', 'expiration_month', ]
	for head in list(train):
	    if head in to_one:
	       # print head
	        start = time.time()
	        one_hot = pd.get_dummies(train[head])
	        one_hot_final = one_hot_final.drop([head], axis=1)
	        for one_head in list(one_hot):
	            one_hot_final[head + '_' + str(one_head)] = one_hot[one_head]
	            to_one += head + '_' + str(one_head)
	       # print 'run time:', time.time() - start, '\n'
	        
	train = one_hot_final
	for head in list(train):
	    if head == 'target':
	        train[head] = train[head].astype(np.uint8)
	    if head in to_one:
	        train[head] = train[head].astype(np.uint8)
	return train
	

def train_and_validate(train, test):
	for col in train.columns:
		if train[col].dtype == object:
			train[col] = train[col].astype('category')
			test[col] = test[col].astype('category')

	print "One hot encoding..."	
	train = one_hot_enc(train)
	test = one_hot_enc(test)
	print 'Done.'
    
    print "Preparing dev set..."
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
	y_pred = lgb_model.predict(X_dev)
	print "dev accuracy: " + str(1 - mean_squared_error(Y_dev, y_pred) ** 0.5)
	return lgb_model, test

def generate_predictions(lgb_model, test):
	print "Generating predictions for test set..."
	X_test = test.drop(['id'], axis=1)
	ids = test['id'].values
	print len(ids)
	predictions = lgb_model.predict(X_test)
	print len(predictions)
	output = pd.DataFrame()
	output['id'] = ids
	output['target'] = predictions
	output.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

if __name__ == "__main__":
	train, test, members, songs, songs_extra = load_data()
	train, test, members = merge_and_fix_data(train, test, songs, songs_extra, members)
	train, test = add_new_features(train, test)
	lgb_model, test = train_and_validate(train, test)
	generate_predictions(lgb_model, test)
