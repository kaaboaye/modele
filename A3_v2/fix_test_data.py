import pickle

with open('test_data.pkl', mode='rb') as file_:
    TEST_DATA = pickle.load(file_)

TEST_DATA['opt'].pop('obj_fun')
TEST_DATA['sopt'].pop('obj_fun')

with open('test_data.pkl', mode='wb') as file_:
    pickle.dump(TEST_DATA, file_, protocol=pickle.HIGHEST_PROTOCOL)
