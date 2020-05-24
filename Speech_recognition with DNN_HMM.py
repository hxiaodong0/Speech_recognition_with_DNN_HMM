
import argparse
import logging
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import torch.utils.data as Data

# neural network related
from sklearn.neural_network import MLPClassifier

def get_data_dict(data):
    data_dict = {}
    for line in data:
        if "[" in line:
            key = line.split()[0]
            mat = []
        elif "]" in line:
            line = line.split(']')[0]
            mat.append([float(x) for x in line.split()])
            data_dict[key]=np.array(mat)
        else:
            mat.append([float(x) for x in line.split()])
    return data_dict


#you can add more functions here if needed

def logsumexp(xs):
    max = np.max(xs)
    ds = xs - max
    return max + np.log(np.exp(ds).sum())

class SingleGauss():
    def __init__(self):
        self.dim = None
        self.stats0 = None
        self.stats1 = None
        self.stats2 = None

        self.mu = None
        self.r = None
        self.logconst = None

    def train(self, data):
        data_mat = np.vstack(data)
        self.dim = data_mat.shape[1]

        self.stats0 = data_mat.shape[0]
        self.stats1 = np.sum(data_mat, axis=0)
        self.stats2 = np.sum(data_mat * data_mat, axis=0)

        # mu = sum o / T
        self.mu = self.stats1 / self.stats0
        # r = (sum o^2 - T mu^2)/T
        self.r = (self.stats2 - self.stats0 * self.mu * self.mu) / self.stats0

        # -0.5 D log(2pi) -0.5 sum (log r_d)
        self.logconst = get_log_const(self.dim, self.r)

    def loglike(self, data_mat):
        stats0 = data_mat.shape[0]
        stats1 = np.sum(data_mat, axis=0)
        stats2 = np.sum(data_mat * data_mat, axis=0)

        # ll = T log_const - 0.5 (stats2 - 2 * stats1 * mu + stats0 * mu^2)/r
        ll = stats0 * self.logconst
        ll += -0.5 * np.sum((stats2 - 2 * stats1 * self.mu + stats0 * self.mu * self.mu) / self.r)

        return ll

def get_log_const(dim, r):
    return - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(r))


class HMM():
    def __init__(self, gauss, nstate=3):
        self.hmm = []
        self.nstate = nstate
        for j in range(nstate):
            self.hmm.append(SingleGauss())

        # init
        self.pi = np.zeros(nstate)
        self.pi[0] = 1.0  # left to right
        # state trans, left to right, consider final state
        self.a = np.zeros((nstate, nstate))
        for j in range(nstate - 1):
            self.a[j, j] = 0.5
            self.a[j, j + 1] = 0.5
        self.a[nstate - 1, nstate - 1] = 1.0
        # init with purturb
        for j in range(nstate):
            self.hmm[j].dim = gauss.dim
            self.hmm[j].mu = gauss.mu
            self.hmm[j].r = gauss.r
            self.hmm[j].logconst = get_log_const(gauss.dim, gauss.r)

        # stats
        self.statsa = None

        # for log(0)
        self.bigminus = -10000000

    def uniform_state(self, data_mat):
        seq = np.zeros(data_mat.shape[0], dtype=np.int)
        interval = int(data_mat.shape[0] / self.nstate)
        seq[:interval] = 0
        prev = interval
        for i in range(1, self.nstate - 1):
            seq[prev:prev + interval] = i
            prev = prev + interval
        seq[prev:] = self.nstate - 1

        return seq

    def forward(self, data_mat):
        # delta T x J
        logalpha = np.zeros((data_mat.shape[0], self.nstate))

        # t = 1 only consider j = 0, as it is left to right
        logalpha[0, 0] = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        for i in range(1, self.nstate):
            logalpha[0, i] = self.bigminus + self.hmm[i].loglike(np.expand_dims(data_mat[0], axis=0))
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [self.hmm[j].loglike(np.expand_dims(data_mat[t], axis=0)) for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logalpha_all = []
                logalpha[t, j] = lls[j]
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logalpha_all.append(loga + logalpha[t - 1, i])
                logalpha[t, j] += logsumexp(np.array(logalpha_all))

        loglike = logsumexp(logalpha[data_mat.shape[0] - 1, :])

        return logalpha, loglike


    def viterbi(self, data_mat):
        # delta T x J
        logdelta = np.zeros((data_mat.shape[0], self.nstate))
        # psi T x J
        psi = np.zeros((data_mat.shape[0], self.nstate))

        # t = 1 only consider j = 0, as it is left to right
        logdelta[0, 0] = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        for i in range(1, self.nstate):
            logdelta[0, i] = self.bigminus + self.hmm[i].loglike(np.expand_dims(data_mat[0], axis=0))
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [self.hmm[j].loglike(np.expand_dims(data_mat[t], axis=0)) for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logdelta_all = []
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logdelta_all.append(loga + logdelta[t - 1, i] + lls[j])
                logdelta[t, j] = np.max(np.array(logdelta_all))
            # get psi
            for j in range(self.nstate):
                psi_all = []
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    psi_all.append(loga + logdelta[t - 1, i])
                psi[t, j] = np.argmax(np.array(psi_all))

        # back tracking
        seq = np.zeros(data_mat.shape[0], dtype=np.int)
        seq[data_mat.shape[0] - 1] = np.argmax(logdelta[data_mat.shape[0] - 1, :])
        for t in range(data_mat.shape[0] - 1)[::-1]:
            seq[t] = psi[t + 1, seq[t + 1]]

        if seq[0] != 0:
            logging.warn("???")

        logging.debug(seq)
        return seq

    def estep_viterbi(self, data, uniform=False):
        # init stats
        for j in range(self.nstate):
            self.hmm[j].stats0 = 0
            self.hmm[j].stats1 = np.zeros(self.hmm[j].dim)
            self.hmm[j].stats2 = np.zeros(self.hmm[j].dim)
        self.statsa = np.zeros((self.nstate, self.nstate))

        data_seq = []
        for data_mat in data:
            # gamma T x J
            gamma = np.zeros((data_mat.shape[0], self.nstate))
            # xi (T-1) x J
            xi = np.zeros((data_mat.shape[0] - 1, self.nstate, self.nstate))

            if uniform is True:
                # uniform state seq
                seq = self.uniform_state(data_mat)
            else:
                # get most likely state seq
                seq = self.viterbi(data_mat)
            data_seq.append(seq)

            # t = 1
            gamma[0, 0] = 1
            for i in range(self.nstate):
                self.hmm[i].stats0 += gamma[0, i]
                self.hmm[i].stats1 += gamma[0, i] * data_mat[0]
                self.hmm[i].stats2 += gamma[0, i] * data_mat[0] * data_mat[0]
            # t > 1
            for t in range(1, data_mat.shape[0]):
                gamma[t, seq[t]] = 1
                for i in range(self.nstate):
                    for j in range(self.nstate):
                        if seq[t - 1] == i and seq[t] == j:
                            xi[t - 1, i, j] = 1

                for i in range(self.nstate):
                    self.hmm[i].stats0 += gamma[t, i]
                    self.hmm[i].stats1 += gamma[t, i] * data_mat[t]
                    self.hmm[i].stats2 += gamma[t, i] * data_mat[t] * data_mat[t]
                    for j in range(self.nstate):
                        self.statsa[i, j] += xi[t - 1, i, j]

        return data_seq

    def mstep(self):
        self.pi = np.zeros(self.nstate)
        self.pi[0] = 1.0

        # state transition (left to right)
        self.a = np.zeros((self.nstate, self.nstate))
        for j in range(self.nstate - 1):
            self.a[j, j] = self.statsa[j, j] / (self.statsa[j, j] + self.statsa[j, j + 1])
            self.a[j, j + 1] = self.statsa[j, j + 1] / (self.statsa[j, j] + self.statsa[j, j + 1])
        self.a[self.nstate - 1, self.nstate - 1] = 1.0

        for j in range(self.nstate):
            # mu = sum o / T
            self.hmm[j].mu = self.hmm[j].stats1 / self.hmm[j].stats0
            # r = (sum o^2 - T mu^2)/T
            self.hmm[j].r = (self.hmm[j].stats2 - self.hmm[j].stats0 * self.hmm[j].mu * self.hmm[j].mu) / self.hmm[
                j].stats0

            # -0.5 D log(2pi) -0.5 sum (log r_d)
            self.hmm[j].logconst = - 0.5 * self.hmm[j].dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(self.hmm[j].r))

        logging.debug("count per digit = %f", sum([self.hmm[j].stats0 for j in range(self.nstate)]))

    def loglike(self, data_mat, seq=None):
        if seq is None:
            seq = self.viterbi(data_mat)

        ll = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        prev_state = 0
        for t in range(1, data_mat.shape[0]):
            state = seq[t]
            ll += self.hmm[state].loglike(np.expand_dims(data_mat[t], axis=0)) + np.log(self.a[prev_state, state])
            prev_state = state
        return ll
    #TODO: Define class variables and member functions (Use from project1)

class NN():
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=200, solver='adam', verbose=True, tol=1e-4,
                            random_state=1,
                            early_stopping=True, validation_fraction=0.1, learning_rate='adaptive')
    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self,x):

        if x.shape[0] == 39 or x.shape[1] == 39:
            x = o_t_gen(digits, x)

        self.prob = self.clf.predict(x)
        self.prid = self.clf.predict_proba(x)

        return self.prob
    def predictprob(self,x):

        if x.shape[0] == 39 or x.shape[1] == 39:
            x = o_t_gen(digits, x)

        self.prid = self.clf.predict_proba(x)

        return self.prid
#mynetwork and training
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size2, num_classes))
        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        out = (self.fc1(x))
        out = (self.fc2(out))
        out = self.fc3(out)

        return out
def train_net(data_total,label_total):
    model = Net(273, 512, 512, 55)
    model.cuda()

    x_train, x_test, y_train, y_test = train_test_split(data_total, label_total, test_size=0.1)

    # x_train = F.normalize(x_train)
    # x_test = F.normalize(x_test)
    y_train = y_train - 1
    y_test = y_test - 1

    train_dataset = Data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )

    train_dataset_test = Data.TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test)
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for e in (range((epochs))):
        running_loss = 0
        running_corrects = 0

        for data in train_loader:
            inputs, label = data
            inputs, labels = Variable(inputs.cuda()), Variable(label.cuda())

            labels = labels.view(labels.shape[0])
            optimizer.zero_grad()
            outputs = model(inputs.float())
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss
            running_corrects += torch.sum(preds == label.cuda())

        print(e, "epochs", running_loss, 'loss', running_corrects, "")
    return model
#end mynetwork and training
class HMMMLP():
    def __init__(self, hmm_model, nstate,k , DNN):

        self.clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=200, solver='adam', verbose=True, tol=1e-4,
                            random_state=1,
                            early_stopping=True, validation_fraction=0.1, learning_rate='adaptive')

        self.hmm = [0,0,0,0,0]
        self.DNN = DNN
        self.ind = k
        for j in range(5):
            self.hmm[j] = hmm_model.hmm[j]
        self.nstate = nstate
        # init
        self.pi = np.zeros(nstate)
        self.pi[0] = 1.0  # left to right
        # state trans, left to right, consider final state
        self.a = np.zeros((nstate, nstate))

        for j in range(nstate - 1):
            self.a[j, j] = 0.5
            self.a[j, j + 1] = 0.5
        self.a[nstate - 1, nstate - 1] = 1.0

        # init with purturb


        # stats
        self.statsa = None

        # for log(0)
        self.bigminus = -10000000
    def loglike(self, data_mat):
        # delta T x J
        data_mat = o_t_gen(digits, data_mat)
        prob = self.DNN.predictprob(data_mat)
        prob = np.log(prob)
        logalpha = np.zeros((data_mat.shape[0], self.nstate))
        index = [[0,1,2,3,4],[5,  6,  7,  8,  9],[10,11, 12, 13, 14],[15,16,
       17, 18, 19],[20,21, 22, 23, 24],[25,26, 27, 28, 29],[30,31, 32, 33,
       34],[35,36, 37, 38, 39],[40,41, 42, 43, 44],[45,46, 47, 48, 49],[50,
       51, 52, 53, 54]]
        # t = 1 only consider j = 0, as it is left to right
        logalpha[0, 0] = np.log(self.pi[0]) + prob[0][(index[self.ind][0])]
        for i in range(1, self.nstate):
            logalpha[0, i] = self.bigminus + prob[0][(index[self.ind][i])]
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [prob[t][(index[self.ind][j])] for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logalpha_all = []
                logalpha[t, j] = lls[j]
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logalpha_all.append(loga + logalpha[t - 1, i])
                logalpha[t, j] += logsumexp(np.array(logalpha_all))
        loglike = logsumexp(logalpha[data_mat.shape[0] - 1, :])

        return loglike






def sg_train(digits, train_data):
    model = {}
    for digit in digits:
        model[digit] = SingleGauss()

    for digit in digits:
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        logging.info("process %d data for digit %s", len(data), digit)
        model[digit].train(data)

    return model


def hmm_train(digits, train_data, sg_model, nstate, niter):
    logging.info("hidden Markov model training, %d states, %d iterations", nstate, niter)

    hmm_model = {}
    for digit in digits:
        hmm_model[digit] = HMM(sg_model[digit], nstate=nstate)
    lst = []
    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        total_count = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)
            datatotal = np.vstack(data)
            # uniform init
            if i == 0:
                data_seq = hmm_model[digit].estep_viterbi(data, uniform=True)
            else:
                data_seq = hmm_model[digit].estep_viterbi(data)

            if i == niter - 1:
                lst.append(data_seq)

            count = sum([len(seq) for seq in data_seq])
            logging.debug("count per digit = %d", count)
            total_count += count

            hmm_model[digit].mstep()

            for data_mat, seq in zip(data, data_seq):
                # total_log_like += hmm_model[digit].loglike(data_mat, seq)
                logalpha, log_like = hmm_model[digit].forward(data_mat)
                total_log_like += log_like

            #TODO: Write HMM training part by calling one or more functions here (Use from project1)

            #TODO: Calculate log likelihood and accumulate the data likelihood
            #      in the variable 'total_log_like' (Use from project1)

        logging.info("log likelihood: %f", total_log_like)
        i += 1
    global data_seq1
    data_seq1 = lst
    return hmm_model




def o_t_gen(digits,data_mat):
    temp_lst = []
    # for i in range(len(digits)):
    #     data = [train_data[id] for id in train_data.keys() if digits[i] in id.split('_')[1]]
    data_mat = np.vstack(data_mat)

    temp_lst.append(data_mat)
    temp_data = np.array(temp_lst[0])
    for i in range(1,len(temp_lst)):
        temp_data = np.vstack((temp_data, temp_lst[i]))  #temp_data is the contacatenated data_set

    cnt1 = 0
    indexs = []
    for i in range(len(temp_data)):
        if i == 0:
            indx = [i,i,i,i,i+1,i+2,i+3]
        elif i == 1:
            indx = [i,i,i,i-1,i+1,i+2,i+3]
        elif i == 2:
            indx = [i-2,i-2,i-1,i,i+1,i+2,i+3]
        elif i == len(temp_data)-3:
            indx = [i-3,i-2,i-1,i,i+1,i+2,i+2]
        elif i == len(temp_data)-2:
            indx = [i-3,i-2,i-1,i,i+1,i+1,i+1]
        elif i == len(temp_data)-1:
            indx = [i-3,i-2,i-1,i,i,i,i]
        else:
            indx = [i-3,i-2,i-1,i,i+1,i+2,i+3]
        indexs +=indx

    o_t = np.array(np.concatenate((temp_data[indexs[0]], temp_data[indexs[1]],temp_data[indexs[2]],temp_data[indexs[3]],temp_data[indexs[4]],temp_data[indexs[5]],temp_data[indexs[6]])),ndmin =2)
# each row 7 indexs;
    # o_t = np.transpose(o_t)

    for i in range(7,len(indexs),7):
        stack = np.array(np.concatenate((temp_data[indexs[i]], temp_data[indexs[i+1]],temp_data[indexs[i+2]],temp_data[indexs[i+3]],temp_data[indexs[i+4]],temp_data[indexs[i+5]],temp_data[indexs[i+6]])),ndmin =2)
        o_t = np.vstack((o_t,stack))


    return o_t


def l_gen(digits , data_mat, digit):

    l= np.array(())
    index = digits.index(digit)  # the index of the input digit
    cnt = 0
    for i in range(len(data_mat)):  # 11
        data_mat[i] = data_mat[i] + 5 * index
    if i == 0:
        l = data_mat[i]
    else:
        l = np.append(l, data_mat[i])
    l = np.array(l, ndmin=2)
    l = np.transpose(l)
    return l


def get_seq(data_total, hmm_model):

    lst = []
    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]
    for digit in digits:
        data = [data_total[id] for id in data_total.keys() if digit in id.split('_')[1]]

        logging.info("process %d data for digit %s get_data", len(data), digit)
        datatotal = np.vstack(data)
            # uniform init
        data_seq = hmm_model[digit].estep_viterbi(data)
        lst.append(data_seq)
    return lst


def mlp_train(digits , train_data , hmm_model, nunits, bsize, nepoch,lr  ):#, nepoch, lr
    data_seq = get_seq(train_data, hmm_model)
    nstate = 5
    niter = 10
    cnt = 0
    for i in range(len(data_seq)):  # 11
        temp = data_seq[i][0] + 5 * i + 1
        for j in range(1, len(data_seq[i])):  # T
            temp = np.append(temp, data_seq[i][j] + 5 * i + 1)
            cnt += 1
        if i == 0:
            l = temp
        else:
            l = np.append(l, temp)

    l = np.array(l, ndmin=2)
    l = np.transpose(l)

    temp_lst = []

    for i in range(len(digits)):
        data = [train_data[id] for id in train_data.keys() if digits[i] in id.split('_')[1]]
        data_mat = np.vstack(data)

        temp_lst.append(data_mat)

    temp_data = np.array(temp_lst[0])
    for i in range(1,len(temp_lst)):
        temp_data = np.vstack((temp_data, temp_lst[i]))  #temp_data is the contacatenated data_set

    o_t = o_t_gen(digits, temp_data)

    # state prior distribution
    unique, counts = np.unique(l, return_counts=True)
    prior_distribution = counts / l.shape[0]
    nunits = (256, 256)
    y = l
    x = o_t
    ylst =[]
    xlst = []
    lst = []

    for i in range(len(digits)):
        ind = np.where(l == 5+i*5)[0][0]
        lst.append(ind)
    lst.append(len(l))

    for i in range(11):
        xlst.append(x[lst[i]:lst[i+1]])
        ylst.append(y[lst[i]:lst[i+1]])

    # y = torch.from_numpy(l)
    # x = torch.from_numpy(o_t)
    #using simple classifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # clf = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=200, solver='adam', verbose=True, tol=1e-4,
    #                      random_state=1,
    #                      early_stopping=True, validation_fraction=0.1, learning_rate='adaptive')
    #
    # clf.fit(x_train, y_train)

    # for i in range(len(xlst)):
    #     clf.fit(xlst[i], ylst[i])

    DNN = NN()
    DNN.train(x_train, y_train)

    y_pred = DNN.predict(x_test)
    kk = accuracy_score(y_test, y_pred)


    mlp_model = {}
    for digit in digits:
        mlp_model[digit] = HMMMLP(hmm_model[digit], nstate=nstate,k = digits.index(digit), DNN = DNN)
        # mlp_model[digit].train(xlst[k], ylst[k])

    lst = []
    i = 0

    # if len(train_data) == 200:
    #     total_count = 0
    #     correct = 0
    #     for key in test_data.keys():
    #         lls = []
    #         for digit in digits:
    #             ll = mlp_model[digit].loglike(test_data[key])
    #             lls.append(ll)
    #         predict = digits[np.argmax(np.array(lls))]
    #         log_like = np.max(np.array(lls))
    #
    #         logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
    #         if predict in key.split('_')[1]:
    #             correct += 1
    #         total_count += 1
    #
    #     logging.info("accuracy: %f", float(correct/total_count * 100))


    return mlp_model





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='test data')
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--nstate', type=int, default=5)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--mode', type=str, default='mlp',
                        choices=['hmm', 'mlp'],
                        help='Type of models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set seed
    np.random.seed(777)

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]

    # read training data

    with open(args.train) as f:  #args.train
        train_data = get_data_dict(f.readlines())

    # for debug
    if args.debug:
        train_data = {key:train_data[key] for key in list(train_data.keys())[:200]}


    # read test data
    with open(args.test) as f:    #args.test
        test_data = get_data_dict(f.readlines())

    with open(args.test) as f:    #args.test
        test_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        test_data = {key:test_data[key] for key in list(test_data.keys())[:200]}

    # Single Gaussian
    sg_model = sg_train(digits, train_data)
    # test the following


    # hmm_model, data_seq = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)

    if args.mode == 'hmm':
        model  = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
    elif args.mode == 'mlp':
        hmm_model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)


        model = mlp_train(digits, train_data, hmm_model, nunits=
(256, 256), bsize=128, nepoch=args.nepoch, lr=args.lr)

    # logging.info("accuracy: %f", float(correct / total_count * 100))

	# #TODO: Modify MLP training function call with appropriate arguments here
    #     model = mlp_train(digits, train_data, hmm_model, nunits=(256, 256), bsize=128, nepoch=args.nepoch, lr=args.lr)

    # test

    total_count = 0
    correct = 0
    for key in test_data.keys():
        lls = [] 
        for digit in digits:
            ll = model[digit].loglike(test_data[key])
            lls.append(ll)
        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))

        logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
        if predict in key.split('_')[1]:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct/total_count * 100))