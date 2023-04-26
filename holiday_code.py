import numpy as np ;
# import requests ;
import os ;
import sys ;
import matplotlib.pyplot as plt;
import tensorflow as tf;

if __name__ == "__main__":

  ## download MNIST if not present in current dir!
  if os.path.exists("./mnist.npz") == False:
    print ("Downloading MNIST...") ;
    fname = 'mnist.npz'
    url = 'http://www.gepperth.net/alexander/downloads/'
    r = requests.get(url+fname)
    open(fname , 'wb').write(r.content)
  
  ## read it into 'traind' and 'trainl'
  data = np.load("mnist.npz")
  traind = data["arr_0"] ;
  trainl = data["arr_2"] ;
  
  if sys.argv[1] == "1":
    # 1. numbers from 10 to 20(included) that are odd
    ex1_1 = [i for i in range(10,21) if i % 2 == 1]
    print("odd num: ", ex1_1)        
    # 2. numbers from 100 to 0(included) that can be divided by 10 (use %, the modulus operator and list comprehension)
    ex1_2 = [i for i in range(100, -1, -1) if i % 10 == 0]
    print("divisible by 10: ", ex1_2)
    
    # 3. numbers from 15 to 1 (included) that can be divided by 3
    ex1_3 = [i for i in range(15, 1, -1) if i%3 == 0]
    print("divisible by 3: ", ex1_3)
    
    # 4. string like “x”, “xx”, “xxx” repeated 10 times. Use the fact that, inPython, a string s multiplied by an integer n results in s repeated n times.
    ex1_4 = ['x'*i for i in range(1, 11)]
    print('x repeatation: ', ex1_4)
    
    # 5. the string ”stringX” repeated 5 times, where X goes from 5 to 0(excluded).Use the builtin function str() to convert numbers to strings and the fact that strings can be concatenated using the ”+” operator
    ex1_5 = ['string'+str(i) for i in range(5, 0, -1)]
    print("string with num: ", ex1_5)
     
    # 6. a list with the items ”1”, 1, 1.0, “one”
    ex1_6 = ["1", 1, 1.0, "one"]
    print(ex1_6)
    
    # 7. all the numbers from 0 to 99 that contain the digit ”5”. You may usethe method find() that all strings possess to look for a substring. If it is found, the start index is returned, otherwise -1.
    ex1_7 = [i for i in range(0, 100) 
             if str(i).find('5') != -1]
    print("contain only 5 in num: ", ex1_7)
     
    
  if sys.argv[1] == "2":
    # a) Create a 1D array with entries from -100 to 0(included) in steps of 2
    ex2_1 = np.arange(-100, 1, 2)
    # ex2_1 = np.linspace(-100, 0, 51, dtype=int)
    print("1D array: ", ex2_1)
    
    # b) Create a 2D with 3 rows and 2 columns, with row entries 1,1..., 2,2,..., 3,3,...
    ex2_2 = np.array([[i,i] for i in range(1,4)])
    print("2D array: ", ex2_2)
    
    # c) Create a 2D with 3 rows and 2 columns that has the value -1 everywhere
    ex2_3 = np.ones([3,2]).astype(int)* -1
    print("2D array with -1 entries: ", ex2_3)
    
    # d) Create a 3D tensor with shape (5,4,3) with random normal entries, with mean 0 and standard deviation 1.
    ex2_4 = np.random.normal(0,1, size=[5,4,3])
    print("3D tensor: ", ex2_4)
    
  if sys.argv[1] == "3":
   
    td = np.ones([50, 5, 5])*np.arange(0, 50, 1).reshape(50, 1,1)
    
    # a) Slice out the 1st sample into an array x and print it!
    x = td[0]
    print("1st sample: ", x)
    
    # Set the 2 lowermost columns of the 10th to -1
    td[9, :, -2:] = -1
    print("lowermost 2 columns: ", td[9])
    
    # c) Print the mean pixel value in the 10th data sample
    ex3_3 = td[9].mean()
    print("mean of sample 10th: ", ex3_3)
    
    # # d) Generate the following variations of the 10th sample and store them in a new variable z:
    ex3_4 = td[9]
    # # just keep every 3rd row
    ex3_4_1 = ex3_4[::3, ::]
    print("3rd row: ", ex3_4_1)
    # # just keep every 3rd column
    ex3_4_2 = ex3_4[::, ::3]
    print("3rd columns: ", ex3_4_2)
    # # invert all rows but not columns
    ex3_4_3 = td[9, ::-1, ::]
    print(ex3_4_3)
    
    #  invert rows but not colums, just keeping every 2th row
    inverted_sample = td[9, ::-1, ::]
    ex3_4_4 = inverted_sample[::2, ::]
    print("invert row, keeping 2th row: ", ex3_4_4)
    #  in-place transform 1+x
    td += 1
    print("in-place 1+x: ",td)
    
  if sys.argv[1] == "4":
    # Reduction 
    td = np.ones([50, 5, 5])*np.arange(0, 50, 1).reshape(50, 1, 1)
    
    
    
    # pixel variance for pixel 0,0 over all samples
    ex4_1 = np.var(td[:, 0, 0].flatten())
    print("variance: ", ex4_1)
    
    # pixel argmax for pixel 0,0 over all samples
    ex4_2 = np.argmax(td[:, 0, 0], axis=0)
    print("argmax: ", ex4_2)
    
    # Compute the “standard deviation image” over all samples
    ex4_3 = np.std(td, axis=0)
    print("standard deviation: ", ex4_3)
    
    #  Compute the row-wise mean (axis=1) over all samples
    ex4_4 = np.mean(td, axis=1)
    print("row-wise mean(axis=1): ", ex4_4)
    
    # Compute the column-wise(axis= 2) mean over all samples
    ex4_5 = np.mean(td, axis=(2))
    print("column-wise mean (axis=0,2): ", ex4_5)
    
  if sys.argv[1] == '5':
    # reduction using 3D array
    td = np.zeros([50, 5, 5]) * np.arange(0,50,1).reshape(50, 1, 1)
    
    # create a 5-element row vector with entries from 1 to 5, and subtract it from all rows of all samples using broadcasting
    new_vec = np.arange(1, 6, 1).reshape(1, 1, 5)
    ex5_1 = td - new_vec
    print("subtract rows: ", ex5_1[0])
    
    # 5-element column vector with entries from 1 to 5, and multiply it with all columns of all samples using broadcasting
    new_vec1 = np.arange(1,6,1).reshape(1, 5, 1)
    ex5_2 = np.multiply(td, new_vec1)
    print("multiply columns: ", ex5_2[1])
    
    #  compute the mean image over all samples, and subtract it from all samples via broadcasting
    ex5_3 = np.mean(td, axis=(0,2)) - td
    print("mean & subtract: ", ex5_3[1])    
    
  if sys.argv[1] == "6":
    # fancy indexing
    example_fancy = np.zeros([5,4])
    example_fancy[2, :] = 3
    indices = [0,2]
    example_fancy = example_fancy[indices, :]
    print("indice 0, 2: ", example_fancy)
    
    # mask indexing
    example_mask = np.zeros([2, 3])
    example_mask[1, :] = 3
    maske = (example_mask == 5)
    print("example of mask indexing",example_mask[maske])
    
    # Fancy indexing and mask indexing
    # create a 20-element vector with entries from 1 to 20, and copy out all elements that are even using mask indexing!
    ex6_1 = np.arange(1, 21, 1)
    maske = (ex6_1 % 2 == 0 )
    print("copy even vectors using mask: ", ex6_1[maske])
    
    # create a 20-element vector with entries from 1 to 20, and copy out elements at positions 1, 5 amd 10 using a single operation!

    ex6_2 = ex6_1[[1,5,10]]
    print("using indices, copy indexs: ", ex6_2)
    
  if sys.argv[1] == "7":
    # Matplotlib
    #  plot the function 1/x between 1 and 5 using 100 support points!

    x = np.linspace(1, 5, 100)
    plt.plot(x, 1/x)
    plt.show()
    
    # generate a scatter plot of the same data as in a)
    x = np.linspace(1, 5, 100)
    y = 1/x
    plt.scatter(x, y)
    plt.show()
    
    #  generate a bar plot of the same data as in a)!
    x = np.linspace(1, 5, 100)
    plt.bar(x, x**2)
    plt.show()
    
    #  plot 1/x and √x together in a single plot, same range as before
    x = np.linspace(1, 5, 100)
    plt.plot(x, 1/x)
    plt.plot(x, np.sqrt(x))
    plt.show()
    
    # generate 100 numbers distributed according to a uniform distribution between 0 and 1, and display their histogram!
    x = np.random.uniform(0, 1, size=100)
    plt.hist(x)
    plt.show()
    
  if sys.argv[1] == "8":
    # MNIST and matplotlib
    # Display samples nr. 5,6 and 7 in a single figure!
    f,ax = plt.subplots(1,3)
    indices = [5,6,7]
    for i, val in enumerate(indices):
      ax[i].imshow(traind[val])
    plt.show()
    
    # Compute the mean pixel value for each image and display all means in a scatter plot!
    x_data = np.linspace(0, 60000, 60000)
    y_data = np.mean(traind, axis=(1,2))
    # print(y_data.size)
    plt.scatter(x_data, y_data)
    plt.show()
    
    #  Copy out all the images whose mean pixel value is > 0.3 and display 3 of them
    f,ax = plt.subplots(1,3)
    maske = traind.mean(axis=(1,2)) > 0.3
    strImg = traind[maske]
    for i in range(0,3):
      ax[i].imshow(strImg[i])
    plt.show()
    
    # Compute the “variance image” over all samples and display it!
    a = traind.mean(axis=0)
    print(a.shape)
    data = traind.var(axis=0).reshape(28,28)
    plt.imshow(data)
    plt.show()
    
    # Copy out 10 random images and display them in a single figure!
    width = 20
    height = 20
    fig = plt.figure(figsize=(6, 6))
    columns = 5
    rows = 2
    for i in range(1, columns*rows +1):
        img = np.random.randint(0,1, size=(height,width))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
    
    # Copy out all samples of class 5 and display 10 of them!
      # /**create maske for class 5 , argmax -> returns all indices w.r.t axis **/
    numerical_classes = trainl.argmax(axis=1)
    maske = (numerical_classes == 5)
    data = traind[maske, :, :]
    f = plt.figure()
    for i in range(0,10):
      f.add_subplot(2,5, i+1)
      plt.imshow(data[i])
    plt.show()
    
  if sys.argv[1] == "11":
    # softmax function Write a python function S(X) which takes an 2D TF tensor and returns the softmax, applied row-wise, as a TF tensor. The function must work for tensors X with an arbitrary number of rows! Print out results for ~x = [−1, −1, 5] and ~x = [1, 1, 2]!
    def softmax(x):
      e = tf.math.exp(x)
      sum_e = tf.reduce_sum(e, axis=1, keepdims=True)
      return e/sum_e
    x1 = tf.constant([[-1., -1, 5]])
    x2 = tf.constant([[1., 1, 2]])
    print(softmax(x1), softmax(x2))
    
  if sys.argv[1] == "12":
    # cross entropy in TF
    # Write a python function MSE(Y,T) which takes an 2D TF tensor and returns the its mean-squared-error (MSE) loss as a TF scalar! Print out results for T = [0, 0, 1] for all rows and Y = ((0.1 0.1 0.8),(0.3 0.3 0.4),(0.8 0.1 0.1)). The function must also work for tensors Y0,T0 that contain only the first two rows of Y and T.
    def MSE(Y,T):
      return tf.reduce_mean(tf.square(Y-T))/Y.shape[0]
    
    Y = tf.constant([[0.1,0.1,0.8], [0.3, 0.3, 0.4], [0.8, 0.1, 0.1]])
    T = tf.constant([[0., 0, 1], [0., 0, 1], [0., 0, 1]])
    # Compute the MSE loss using the MSE function
    loss = MSE(Y, T)

    # Print the result
    print("MSE Loss:", loss)
    T0 = T[:2]
    Y0 = Y[:2]

    # Compute the MSE loss for the first two rows using the MSE function
    loss0 = MSE(Y0, T0)

    # Print the result
    print("MSE Loss (First Two Rows):", loss0)
   
   
    
    # def cross_entropy(y,t):
    #   log_y = tf.math.log(y)
    #   reduce_y_t = tf.reduce_sum(log_y, axis=1, keepdims=True)
    #   return reduce_y_t * t
    
    # y = tf.constant([[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.8, 0.1, 0.1]])
    # t = tf.constant([[0. , 0, 1]])
    # print(cross_entropy(y, t))
      
    
      
      
  if sys.argv[1] == "20":
  #  tf gradients
    # Let f(~x) = P3 i=1 ixi.
# a) Implement this function in TF and compute its output for the inputs ~a1 = (1, 2, 3)T and ~a2 = (2, 0, 2)T. Hint: use a tf function to compute the sum!
# b) Use TF to compute and display the value of ∇~ f, evaluated for ~x = ~a1
# c) Use TF to compute and display the value of ∂f∂x1, evaluated for ~x = ~a2
    def f(x):
      i = tf.range(1., x.shape[0]+1, 1)
      print(len(i))
      exp = tf.reduce_sum(i*x)
      return exp
    a1 = tf.constant([1., 5, 3])  # decimal point to have float dtype
    a2 = tf.constant([2., 0, 2])  # for a1 and a2
    print("20a", f(a1), f(a2))
    
    with tf.GradientTape(persistent=True) as g:
      g.watch(a1)
      g.watch(a2)
      y1 = f(a1)
      y2 = f(a2)
      
    # Use TF to compute and display the value of ∇~ f, evaluated for ~x = ~a1
    # x = g.gradient(y1, a1)
    print("gradient w.r.t a1: ", g.gradient(y1, a1))
    # x = g.gradient(y2, a2)
    print("gradient w.r.t a2: ", g.gradient(y1, a1))
    
#     dy_dx_a1 = g.gradient(y1, a1)
#     dy_dx_a2 = g.gradient(y2, a2)
      
#     print(dy_dx_a1, dy_dx_a2)

if sys.argv[1] == "21":
  # x = tf.constant([[1,2,3], [4,5,6]])
  # print(tf.transpose(x))
  
  def f(x):
    i = tf.range(1, 4)
    x_pow = tf.pow(x, i)
    return tf.reduce_sum(x_pow)
  
  x1 = tf.Variable(1.0)
  x2 = tf.Variable(2.0)
  x3 = tf.Variable(3.0)
  with tf.GradientTape() as tape:
    y1 = f(x1)
    y2 = f(x2)
    y3 = f(x3)

  g_x = tape.gradient(y, x)  # g(x) = dy/dx
  print(g_x)
  
if sys.argv[1] == "22":
  #  *** Advanced numpy ***
  
  # a) Give a code snippet that create two random 1D arrays of length 20, with integer entries between 0 and 3 (included). Then, the code should compute the confusion matrix from these two vectors
      
  # Define the two arrays
  arr1 = np.random.randint(4, size=20)
  arr2 = np.random.randint(4, size=20)

  # Define the confusion matrix as a 2D array of zeros
  confusion_matrix = np.zeros((4, 4), dtype=int)
  
  # Loop through the arrays and update the confusion matrix
  for i in range(len(arr1)):
      true_class = arr1[i]
      predicted_class = arr2[i]
      confusion_matrix[true_class][predicted_class] += 1

  # Print the resulting confusion matrix
  print("confusion matrix: ", confusion_matrix)
  
  # b) Give a code snippet that generates two 1D arrays with values from 0 to 19(included) in ascending order. Then, the code should shuffle both arrays such that same positions contain same values after shuffling (like you would shuffle train data and labels).
  y = np.arange(20, dtype=int)
  t = np.arange(20, dtype=int)
  print("output value: ", y,"\n target value: ", t)
  
  p = np.random.permutation(len(t))
  print("\n p: ", p)
  
  t, y = t[p], y[p]
  print("After shuffle \n output Value: ", y, "\n target matrix: ", t)
  
  # c) Give a code snippet that creates a 1D array with random values from 0 to 9 (included). Then, interpret this array as scalar targets and create a one-hot representation for them, assuming 10 classes.
  a = np.random.randint(0, 10, size=20, dtype=int)
  print("1D array: ", a)
  nb_class = 10
  one_hot_matrix = np.eye(10)[a].astype(int)
  print("one hot matrix array: \n", one_hot_matrix)
  
if sys.argv[1] == "23":
  # Affine Layer
  # Given a 2D TF tensor ‘X’ of shape (100,20): give a snippet that performs an affine layer transformation on X. Include the creation of all required TF variables!
  
  # Define the input tensor
  X = tf.constant(tf.random.normal([100,20]))
 
  # Create the weight and bias variables for the affine layer
  W = tf.Variable(tf.random.normal([20, 10]), name='W')
  b = tf.Variable(tf.zeros([10]), name='b')
  
  output = tf.matmul(X, W) + b  
  print("Affine Layer: \n", output)
  
if sys.argv[1] == "24":
  # Assuming a function f(X, W) that performs some computation on ‘X’ using the TF tensor ‘W’: give a code snippet that computes the gradient of f w.r.t. W!

  # Define the function f(X, W)
  def f(X, W):
      # Some computation on X using W
      Y = tf.matmul(X, W)
      # Return the result
      return Y

  # Define the inputs
  X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  print(X)
  
  W = tf.Variable([[0.5, 1.0], [2.0, 3.0]])
  print(W)
  # Compute the gradient of f w.r.t. W
  with tf.GradientTape() as tape:
      Y = f(X, W)
  grad_W = tape.gradient(Y, W)

  # Print the result
  print(grad_W)
  
if sys.argv[1] == "25":
  # create 2D  random array
  X = np.array([[i,i] for i in range(1,4)])
 
  # a) Give a code snippet that prints the number of columns in X that have a sum bigger than 3.0
  maske = (np.sum(X, axis=0) > 3.0)
  print(X[maske[:, 0]])
  
  
  # ******** practise *******
  # 1D
  print("1D: ", np.arange(0,10,1))
    
    # 2D
  print("2D: ", np.array([[i, i] for i in range(0, 10)]))
    
    # 3D
  print("3D: ", np.random.normal(1, 10,[3,2,2]).astype(int))
    

  
  
  
  

  
  
      

