import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class MLP:
    def __init__(self, feature_num, hidden_layer, output_num=2, initialize_weight = None):
        self.feature_num = feature_num
        self.hidden_layer = hidden_layer
        self.output_num = output_num

        # Initialize weights and biases for all layers
        self.W_l = []
        self.b_l = []

        var0 = 2./(self.hidden_layer + self.feature_num)
        self.W_l.append(np.random.randn(self.feature_num,self.hidden_layer)*np.sqrt(var0)) #W input
        self.b_l.append(np.zeros(self.hidden_layer))#b input

        var1 = 2./(self.hidden_layer + self.hidden_layer)
        self.W_l.append(np.random.randn(self.hidden_layer, self.hidden_layer)*np.sqrt(var1))#W1
        self.b_l.append(np.zeros(self.hidden_layer))#b1
        self.W_l.append(np.random.randn(self.hidden_layer, self.hidden_layer)*np.sqrt(var1))#W2
        self.b_l.append(np.zeros(self.hidden_layer))#b2

        var2 = 2./(self.output_num + self.hidden_layer)
        self.W_l.append(np.random.randn(self.hidden_layer, self.output_num)*np.sqrt(var2))#W output
        self.b_l.append(np.zeros(self.output_num))#b output


        if initialize_weight is not None:
            # Initialize weights and biases for all layers
            self.W_l = []
            self.b_l = []

            var0 = 2./(self.hidden_layer + self.feature_num)
            weight_tmp = np.tile(initialize_weight, (self.hidden_layer, 1)).T + np.random.randn(self.feature_num,self.hidden_layer)*np.sqrt(var0)
            max_weight = np.max(weight_tmp)
            self.W_l.append(abs(weight_tmp/max_weight)) #W input
            self.b_l.append(np.zeros(self.hidden_layer)) #b input

            var1 = 2./(self.hidden_layer + self.hidden_layer)
            self.W_l.append(np.random.randn(self.hidden_layer, self.hidden_layer)*np.sqrt(var1))#W1
            self.b_l.append(np.zeros(self.hidden_layer))#b1
            self.W_l.append(np.random.randn(self.hidden_layer, self.hidden_layer)*np.sqrt(var1))#W2
            self.b_l.append(np.zeros(self.hidden_layer))#b2

            var2 = 2./(self.output_num + self.hidden_layer)
            self.W_l.append(np.random.randn(self.hidden_layer, self.output_num)*np.sqrt(var2))#W output
            self.b_l.append(np.zeros(self.output_num))#b output

    # Define the activation functions
    def softmax(self, x):
        """
        Compute softmax values for each row of x in a numerically stable way.
        """
        # Subtract the maximum value in each row to avoid overflow
        max_x = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - max_x)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def softplus(self, x, beta=1, threshold=20): 
        """
        Compute softplus values for each row of x in a numerically stable way.
        Arguments:
        x: Input array
        beta: Parameter, default:1
        threshold: Threshold for reverting to linear function, default:20
        """

        return np.where(beta*x < threshold, np.log(1 + np.exp(beta*x))/beta, x)

    # Define the loss function as KL divergence
    def kl_divergence(self, y_true, y_pred):
        """
        Compute kl_divergence in a numerically stable way.

        Arguments:
        y_true: y dataset(N,2)
        y_pred: prediction on y(N,2)

        Returns:
        kl_divergence: scalar
        """
        # clip values to avoid invalid value in log(0)
        eps = np.finfo(float).eps 
        y_true = np.maximum(y_true, eps) 
        y_pred = np.maximum(y_pred, eps)
        return np.sum(y_true * np.log(y_true / y_pred))


    def dense(self, x, W, b):
        """
        x: K x h_in array of inputs
        W: h_in x h_out array for kernel matrix parameters
        b: Length h_out 1-D array for bias parameters
        returns: K x h_out output array 
        """

        h = b + x @ W  ## <-- pre-activations
        return h

    def forward(self, X):
        """    
        Arguments:
        W_l: list of h_in x h_out array for kernel matrix parameters
        b_l: list of bias parameters
        X: input dataset of (N,D)

        Returns:
        a_l: list of array of hidden layer pre-activations
        h_l: list of array of hidden layer post-activations
        """

        #Initialize empty list
        h_l = []
        a_l = [] 

        h_l.append(X)

        a = self.dense(X,self.W_l[0],self.b_l[0]) 
        h = (a)
        a_l.append(a)
        h_l.append(h)

        a = self.dense(h,self.W_l[1],self.b_l[1]) 
        h = self.softplus(a)
        a_l.append(a)
        h_l.append(h)

        a = self.dense(h,self.W_l[2],self.b_l[2]) 
        h = self.softplus(a)
        a_l.append(a)
        h_l.append(h)

        #the last output layer
        a = self.dense(h,self.W_l[3],self.b_l[3]) 
        h = self.softmax(a)
        a_l.append(a)
        h_l.append(h)

        return a_l,h_l

    #derivative of softplus
    def activation_derivative(self, x, threshold=500):
        """
        x: K x width array of hidden layer pre-activations
        returns: K x width array of diagonal elements  
        """
        x = np.clip(x,-threshold, threshold)
        return 1 / (1 + np.exp(-x))

    def softmax_derivative(self, x):
        s = self.softmax(x).reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)


    def output_error(self, y, a):
        """
        y: K x 2 array of data outputs
        a: K x 2 array of output pre-activations
        returns: 
        delta: K x 2 array of output errors 
        """
        d_lh = -np.divide(y,self.softmax(a))


        delta = np.zeros(a.shape)
        for n in range(a.shape[0]):
            d_ha = self.softmax_derivative(a[n,:])
            delta[n,:] = d_lh[n,:] @ d_ha

        return delta

    def backpropagate(self, delta, W, a):
        """
        delta: K x 2 array of output errors
        W: width x 2 array
        a: K x width array of hidden layer pre-activations
        returns: K x width array of hidden layer errors
        # K here is the minibatch size, hence we are applying the backpropagation formula 
        # for each sample in the mini-batch (this is why we use the element-wise product)
        """

        return self.activation_derivative(a) * (delta @ (W.T))

    def sgd(self, a_l, h_l, x_batch, y_batch, learning_rate):
        """
        Updates gradient using Stochastic Gradient Descent 
        Arguments:
        a_l: list of array of hidden layer pre-activations
        h_l: list of array of hidden layer post-activations
        W_l: list of h_in x h_out array for kernel matrix parameters
        b_l: list of bias parameters
        x_batch: batch of input dataset of (batchsize,784)
        y_batch: batch of label dataset of (batchsize,)
        learning rate: parameter controlling the speed of algorithm

        Returns:
        W_l: updated list of h_in x h_out array for kernel matrix parameters
        b_l: updated list of bias parameters
        """

        delta = [[],[],[],[]]

        # output error
        y_category = np.eye(2)[y_batch]
        delta[3] = self.output_error(y_category,a_l[-1])

        delta[2] = self.backpropagate(delta[3],self.W_l[3],a_l[2])
        delta[1] = self.backpropagate(delta[2],self.W_l[2],a_l[1])
        delta[0] = self.backpropagate(delta[1],self.W_l[1],a_l[0])

        batchsize = len(y_batch)

        for i in range(4):
            grad_W = (h_l[i].T)@delta[i]/batchsize
            grad_b = np.mean(delta[i], axis=0)

            #update new W and b
            self.W_l[i] = self.W_l[i] - learning_rate*grad_W
            self.b_l[i] = self.b_l[i] - learning_rate*grad_b


    def batch_generator(self, X, y, batchsize=128):
        """
        Generator for dividing datasets into batches for sgd use.
        """
        np.random.seed(1)
        N = X.shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)
        for start_idx in range(0, N, batchsize):
            end_idx = min(start_idx + batchsize, N)
            idx = indices[start_idx:end_idx]
            yield X[idx], y[idx]           



    def accuracy(self, y_pred, y):
        """
        Arguments:
        y_pred: prediction on y(N,)
        y: y dataset(N,)

        Returns:
        accuracy: scalar
        """

        return np.mean(y_pred==y)

    def mlp(self, X_train, y_train, X_test, y_test, epochs=40, learning_rate = 0.01, batchsize = 128, width=60, printval=None, returna=None):
        """
        Train and test on dataset using MLP with sgd, and KL divergence as loss function.
        Arguments:
        X_train: input training dataset of (N,784)
        y_train: training label dataset of (N,)
        X_test: input testing dataset of (N',784)
        y_test: testing label dataset of (N',)
        epochs: number of epochs used to train the model
        learning rate: parameter controlling the speed of algorithm
        batchsize: parameter controlling size of the batch
        width: number of hidden neurons at each hidden layer
        printval: True for printing outcomes every epoch
        returna: True for returning activation of the first hidden layer

        Returns:
        loss_train:array of training loss for each epoch (epochs,)
        acc_train:array of training accuracy for each epoch (epochs,)
        loss_test:array of testing loss for each epoch (epochs,)
        acc_test:array of testing accuracy for each epoch (epochs,)
        """

        #initialize
        loss_train = np.zeros(epochs)
        acc_train = np.zeros(epochs)
        loss_test = np.zeros(epochs)
        acc_test = np.zeros(epochs)
        auc_test = np.zeros(epochs)
        activation_l = []

        for i in range(epochs):
            for x_batch,y_batch in self.batch_generator(X_train,y_train,batchsize):

                a_l,h_l = self.forward(x_batch)

                #update all of the weight function and bias term using stochastic gradient decent
                self.sgd(a_l, h_l, x_batch, y_batch, learning_rate)

            #training dataset
            _,h_train = self.forward(X_train)
            y_pred_train = h_train[-1]
            loss_train[i] = self.kl_divergence(np.eye(2)[y_train],y_pred_train)/len(y_train)#divide by the size of the dataset to ensure comparability
            acc_train[i] = self.accuracy(y_pred_train.argmax(axis=1), y_train)

            _,h_test = self.forward(X_test)
            y_pred_test = h_test[-1]
            loss_test[i] = self.kl_divergence(np.eye(2)[y_test],y_pred_test)/len(y_test)#divide by the size of the dataset to ensure comparability
            acc_test[i] = self.accuracy(y_pred_test.argmax(axis=1), y_test)
            auc_test[i] = roc_auc_score(y_test, y_pred_test[:,1])
            if printval: #if printval, print the result 
                print(f"Epoch: {i+1:03} | Train Loss: {loss_train[i]:.04} | Train Accuracy: {acc_train[i]:.04}| Test Loss: {loss_test[i]:.04} | Test Accuracy: {acc_test[i]:.04} | Test AUC: {auc_test[i]:.04}")

            #store the first activation on the test data
            activation = h_test[1]
            activation_l.append(activation)

        if returna:
            return activation_l  
        return loss_train ,acc_train,loss_test,acc_test,auc_test



def train(X_train, y_train, X_test, y_test, epochs_val = 40):
    num_lr = 4
    lr_l = [5e-3,1e-2,0.05,0.1]
    loss_train_l = []
    loss_test_l = []
    acc_train_l = []
    acc_test_l = []
    auc_test_l = []


    for i in range(num_lr): 
        model = MLP(X_train.shape[1], 64 , 2)
        loss_train ,acc_train,loss_test,acc_test, auc_test = model.mlp(X_train,y_train,X_test,y_test,learning_rate=lr_l[i],printval=True,epochs=epochs_val)

        #store the results
        loss_train_l.append(loss_train)
        loss_test_l.append(loss_test)
        acc_train_l.append(acc_train)
        acc_test_l.append(acc_test)
        auc_test_l.append(auc_test)

    #plot final values only
    plt.figure(figsize=(9,6))
    plt.plot(np.log10(lr_l),np.array(loss_train_l)[:,-1],label='final train loss')
    plt.plot(np.log10(lr_l),np.array(loss_test_l)[:,-1],label='final test loss')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xticks(np.log10(lr_l),lr_l)
    plt.xlabel("Learning rate",size=15)
    plt.ylabel("Loss",size=15);
    plt.title("Final loss of MLP for different learning rate",size=15)



def cv_result(X_train, y_train, num_folds = 5, epochs_val = 40):
    #define cross validation
    kf = KFold(n_splits = num_folds, shuffle = True)
    cv_results = []

    #cross validation for learning rate, max_depth tree 
    learning_rates = [0.1, 0.05]

    for learning_rate in learning_rates:

        fold_scores = []
        print("start training")
        for train_index, val_index in kf.split(X_train):
            # Split data into train and validation sets for each fold
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # Train the mlp model
            model = MLP(X_train.shape[1], 64 , 2)
            loss_train ,acc_train,loss_test,acc_test, auc_test = model.mlp(X_tr,y_tr,X_val,y_val,learning_rate=learning_rate,printval=True,epochs=epochs_val)

            # Evaluate the model auc on the validation set
            fold_scores.append(auc_test[-1])

        # Calculate the average score across all folds for the current learning rate
        avg_score = np.mean(fold_scores)
        cv_results.append((learning_rate, avg_score))
        print(f'learning rate: {learning_rate} is trained with auc {avg_score}')

    return cv_results

def best_parameter(cv_results):
    metric_val = [cv_results[i][-1] for i in range(len(cv_results))]
    largest_ind = metric_val.index(max(metric_val))
    return cv_results[largest_ind]

def train_using_cv(X_train, y_train, X_test, y_test, epochs_val = 40):
    cv_res = cv_result(X_train, y_train)
    best_pair = self.best_parameter(cv_res)

    #redefine the model to train for the optimal lr
    model = MLP(X_train.shape[1], 64 , 2)
    loss_train ,acc_train,loss_test,acc_test, auc_test = model.mlp(X_train,y_train,X_test,y_test,learning_rate=best_pair[0],printval=True,epochs=epochs_val)
    return auc_test[-1], best_pair[0]




from nbsdk import get_table, get_pandas
table_new = get_table('e92229cd-c6a7-4389-8d31-31d3d9970478/churn_feat_diff_final.table')
df_new = table_new.to_pandas()
df_new['churn_100'] = df_new['churn_100'].astype(np.int64)
df_new = df_new.fillna(-1)
df_new.loc[:,"ta_diff"]=df_new.loc[:,"ta_diff"].astype("float")
df_new.loc[:,"ta_diff"]=df_new.loc[:,"tc_diff"].astype("float")


X_tmp_100_new,y_100_new = df_new.iloc[:,1:-4], df_new.iloc[:,-4]
y_100_new =  np.array(y_100_new)
X_100_new = pd.get_dummies(X_tmp_100_new)
X_train_100_new, X_test_100_new, y_train_100_new, y_test_100_new = train_test_split(X_100_new,y_100_new,test_size = 0.25, random_state = 40)
scaler = preprocessing.StandardScaler().fit(X_train_100_new)
X_train_100_new = scaler.transform(X_train_100_new)
X_test_100_new = scaler.transform(X_test_100_new)


best_lr_auc, best_lr = train_using_cv(X_train_100_new,y_train_100_new,X_test_100_new,y_test_100_new)



model = MLP(X_train_100_new.shape[1], 64 , 2)
loss_train ,acc_train,loss_test,acc_test, auc_test = model.mlp(X_train_100_new,y_train_100_new, X_test_100_new, y_test_100_new,learning_rate=0.01,printval=True,epochs=40)