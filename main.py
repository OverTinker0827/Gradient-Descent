class LinearRegression:

  def __init__(self,predictors,outputs):

      weights=np.random.rand(predictors,outputs)
      biases=np.random.rand(1,outputs)
      self.params=[weights,biases]
  def forward(self,params,x):
    return x@params[0]+params[1]
  def mse(self,actual,predicted):
      return np.mean((actual-predicted)**2)

  def mse_grad(self,actual,predicted):
     return predicted-actual
  def backward(self,params,x,lr,grad):
    w_grad=(x.T/x.shape[0])@grad
    b_grad=np.mean(grad,axis=0)
    params[0]-=lr*w_grad
    params[1]-=lr*b_grad
    return params
  def fit(self,train_x,train_y,valid_x,valid_y,lr,epochs,freq):

      params=self.params



      for epoch in range(epochs):
          prediction=self.forward(params,train_x)
          grad=self.mse_grad(train_y,prediction)

          params=self.backward(params,train_x,lr,grad)
          if epoch%freq==0:
             prediction=self.forward(params,valid_x)
             valid_loss=self.mse(valid_y,prediction)
             print(f"Epoch {epoch} valid loss {valid_loss}")
      self.params=params
  def predict(self,test_x):
    return self.forward(self.params,test_x)
  
  

  