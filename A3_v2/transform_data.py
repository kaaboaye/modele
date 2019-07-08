



    # test = {}
    # test_mat = scipy.io.loadmat('test.mat')
    # test['sigm'] = {}
    # test['cost'] = {}
    # test['opt'] = {}
    # test['sopt'] = {}
    # test['rcost'] = {}
    # test['pred'] = {}
    # test['fm'] = {}
    # test['ms'] = {}
    #
    # sarg = test_mat['test']['sigm'][0][0]['arg'][0][0][0][0]
    # sval = test_mat['test']['sigm'][0][0]['val'][0][0][0][0]
    #
    # test['sigm']['arg'] = np.array([[sarg],[sarg]])
    # test['sigm']['val'] = np.array([[sval],[sval]])
    #
    # test['cost']['x_train'] = np.array(test_mat['test']['cost'][0][0]['xTrain'][0][0])
    # test['cost']['y_train'] = np.array(test_mat['test']['cost'][0][0]['yTrain'][0][0])
    # test['cost']['w'] = np.array(test_mat['test']['cost'][0][0]['w'][0][0])
    # test['cost']['L'] = np.array(test_mat['test']['cost'][0][0]['L'][0][0][0][0])
    # test['cost']['grad'] = np.array(test_mat['test']['cost'][0][0]['grad'][0][0])
    #
    # test['opt']['x_train'] = np.array(test_mat['test']['opt'][0][0]['xTrain'][0][0])
    # test['opt']['y_train'] = np.array(test_mat['test']['opt'][0][0]['yTrain'][0][0])
    # test['opt']['w0'] = np.array(test_mat['test']['opt'][0][0]['w0'][0][0])
    # test['opt']['step'] = np.array(test_mat['test']['opt'][0][0]['step'][0][0][0][0])
    # test['opt']['w'] = np.array(test_mat['test']['opt'][0][0]['w'][0][0])
    # test['opt']['func_values'] = np.array(test_mat['test']['opt'][0][0]['func_values'][0][0])
    # test['opt']['epochs'] = np.array(test_mat['test']['opt'][0][0]['epochs'][0][0][0][0])
    #
    # test['sopt']['x_train'] = np.array(test_mat['test']['sopt'][0][0]['xTrain'][0][0])
    # test['sopt']['y_train'] = np.array(test_mat['test']['sopt'][0][0]['yTrain'][0][0])
    # test['sopt']['w0'] = np.array(test_mat['test']['sopt'][0][0]['w0'][0][0])
    # test['sopt']['step'] = np.array(test_mat['test']['sopt'][0][0]['step'][0][0][0][0])
    # test['sopt']['w'] = np.array(test_mat['test']['sopt'][0][0]['w'][0][0])
    # test['sopt']['func_values'] = np.array(test_mat['test']['sopt'][0][0]['func_values'][0][0])
    # test['sopt']['epochs'] = np.array(test_mat['test']['sopt'][0][0]['epochs'][0][0][0][0])
    # test['sopt']['mini_batch'] = np.array(test_mat['test']['sopt'][0][0]['mini_batch'][0][0][0][0])
    #
    # test['rcost']['x_train'] = np.array(test_mat['test']['rcost'][0][0]['xTrain'][0][0])
    # test['rcost']['y_train'] = np.array(test_mat['test']['rcost'][0][0]['yTrain'][0][0])
    # test['rcost']['w'] = np.array(test_mat['test']['rcost'][0][0]['w'][0][0])
    # test['rcost']['L'] = np.array(test_mat['test']['rcost'][0][0]['L'][0][0][0][0])
    # test['rcost']['grad'] = np.array(test_mat['test']['rcost'][0][0]['grad'][0][0])
    # test['rcost']['lambda'] = np.array(test_mat['test']['rcost'][0][0]['lambda'][0][0][0][0])
    #
    # test['pred']['x'] = np.array(test_mat['test']['pred'][0][0]['X'][0][0])
    # test['pred']['w'] = np.array(test_mat['test']['pred'][0][0]['w'][0][0])
    # test['pred']['y'] = np.array(test_mat['test']['pred'][0][0]['y'][0][0])
    # test['pred']['theta'] = np.array(test_mat['test']['pred'][0][0]['theta'][0][0][0][0])
    #
    # test['fm']['y'] = np.array(test_mat['test']['fm'][0][0]['y'][0][0])
    # test['fm']['y_pred'] = np.array(test_mat['test']['fm'][0][0]['yPred'][0][0])
    # test['fm']['f'] = np.array(test_mat['test']['fm'][0][0]['f'][0][0][0][0])
    #
    # test['ms']['x_train'] = np.array(test_mat['test']['ms'][0][0]['xTrain'][0][0])
    # test['ms']['y_train'] = np.array(test_mat['test']['ms'][0][0]['yTrain'][0][0])
    # test['ms']['w0'] = np.array(test_mat['test']['ms'][0][0]['w0'][0][0])
    # test['ms']['w'] = np.array(test_mat['test']['ms'][0][0]['w'][0][0])
    # test['ms']['thetas'] = np.array(test_mat['test']['ms'][0][0]['thetas'][0][0])
    # test['ms']['lambdas'] = np.array(test_mat['test']['ms'][0][0]['lambdas'][0][0])
    # test['ms']['x_val'] = np.array(test_mat['test']['ms'][0][0]['xVal'][0][0])
    # test['ms']['y_val'] = np.array(test_mat['test']['ms'][0][0]['yVal'][0][0])
    # test['ms']['F'] = np.array(test_mat['test']['ms'][0][0]['F'][0][0])
    # test['ms']['step'] = np.array(test_mat['test']['ms'][0][0]['step'][0][0][0][0])
    # test['ms']['epochs'] = np.array(test_mat['test']['ms'][0][0]['epochs'][0][0][0][0])
    # test['ms']['mini_batch'] = np.array(test_mat['test']['ms'][0][0]['mini_batch'][0][0][0][0])
    # test['ms']['theta'] = np.array(test_mat['test']['ms'][0][0]['theta'][0][0][0][0])
    # test['ms']['lambda'] = np.array(test_mat['test']['ms'][0][0]['lambda'][0][0][0][0])
    #
    #
    # pickle.dump(test,open('test_data.pkl','wb'))

    test = pickle.load(open('test_data.pkl','rb'))