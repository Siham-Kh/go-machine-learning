# Machine-learning: Deep Learning Neural Network (DNN)

This is a Go implementation of DNN neural network. It is currently supporting:

- Multiple layers with several neurons each
- Activation functions: sigmoid, Relu
- Optimizer: Gradient Descent (GD), Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent (MGD)
- Classification modes: multi-label, binary
- Bias node


1. Input data scaling:

This step eliminates the measurement units of daya and enables an easy comparison of different inputs. Two common ways to scale the data are:

- Normalization: Scale data to have values between 0 and 1. This is done using the U-score transformation:

                Xnew = (Xold-Xmin)/(Xmax-Xmin)

                Xmin: sample minimum
                Xmax: sample maximum

- Standarization: Transform data to have a mean of 0 and a standard derivation of 1. This is done using the Z-score transformation:

                Xnew = (Xold-Xbar)/S 

                S: sampe standard deviation
                Xbar: sample mean

P.S:
- No scaling is applied to Target data (output)
- when prediction in regression mode, scaling must be applied to output


2.  Weights initialization:

If the activation function is set to Sigmoid, Xavier initialization is chosen by default. 

- Uniform distribution: (Min, Max) = (-r, r) where r = sqrt(6/(in+out))
- Normal Distribution: (Mu, sigma) = (0, r) where r = sqrt(2/(in+out))


If the activation function is set to Relu or its variants, He initialization is chosen by default.

- Uniform distribution: (min, max)= (-r, r) where r = sqrt(2)*sqrt(6/(in+out))
- Normal distribution: (Mu, sigma)= (0, r) where r = sqrt(2)*sqrt(2/(in+out))


If the activation is set to Tnh, He initialization is chosen by default.

- Uniform distribution: (min, max)= (-r, r) where r = sqrt(6/(in+out))
- Normal distribution: (Mu, sigma)= (0, r) where r = sqrt(2/(in+out))


3.  Activation function:

The user can choose between Relu and its variants or sigmoid


4. Optimization algorithm:
 
This library implements two variants of gradient descent:
    - Batch gradent descent to be used in the case of a small data set
    - Stochastic gradient descent, to be used in the case of a large data set
    - Mini Batch gradient descent, to be used in the case of a medium data set


5. Accuracy parameter for early stopping:

           Accuracy = Numerof correct predictions / Total number of predictions
