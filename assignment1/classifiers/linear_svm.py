from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.(3073,100)
    - X: A numpy array of shape (N, D) containing a minibatch of data.(m,3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.(m,)
    - reg: (float) regularization strength 正则项(sum(wi**2))**0.5

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] #10
    num_train = X.shape[0] #500
    loss = 0.0 #由于每张照片有10-1=9个margins，m张照片共有9m个margins，可直接sum
    
    for i in range(num_train): #500
        scores = X[i].dot(W) #计算出第一个照片的10组scores，形成(m,)
        correct_class_score = scores[y[i]] #找出正确label对应的scorei
        for j in range(num_classes): #循环其余9个分数并计算margin
            if j == y[i]: #跳过自身
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin                
                dW[:,j] += X[i]*1.0
                dW[:,y[i]] -= X[i]*1.0

        

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train #取m个loss的均值

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #L2_reg在最终loss上增加,本质是所有w**2的和；对应元素乘
    dW /= num_train #每个Wj都叠加了m组X:由于Loss是均值所以/m
    dW += reg * W * 2 #加上reg导数:W的2倍再*超参数reg
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg): 
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0 #(m,10)
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #计算scores
    scores = X.dot(W)
    #计算margin，取其大于0;同时正确label自身设为0
    margins = np.maximum(0,scores - scores[range(X.shape[0]),y].reshape(-1,1) + 1)
    margins[range(X.shape[0]),y] = 0
    #加合再/m求总L
    loss += np.sum(margins) / X.shape[0]
    #加reg项：w平方的总和*reg
    loss += reg * np.sum(W*W)
    #引出ds使得当对应位置的margin>0则ds内取1；同时改变正确score处为每组样本margin不为0的个数n*-1
    dS = np.zeros_like(margins)
    idx = np.where(margins > 0)
    dS[idx] = 1
    dS[range(X.shape[0]),y] = -1 * np.sum(dS,axis=1)
    #此后得到ds的每行即为：ds=1时，对应dwj；ds取-n时，对于dwy
    dW = X.T.dot(dS)
    dW /= X.shape[0]


    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
