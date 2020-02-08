package main

import (
	"Machine-learning/nn"
	"fmt"
	"log"
	"time"

	"golang.org/x/exp/rand"

)

func main() {

	rand.Seed(uint64(time.Now().UTC().UnixNano()))

	activationFunction := 0 /* 0 for sigmoid ,1 for Relu */
	inputFields := 256
	targetFields := 9
	epochs := 10000000000000
	learningRate := 0.01
	numNeuronsPerLayer := []int{25, targetFields}
	numLayers := len(numNeuronsPerLayer)
	Scale := 0           // 0 for nomalization, 1 for standarizatioon
	distribution := true // true = normal, false = uniform
	pval := false        // activate the validation set (default is 0.1)

	network := nn.BuildNetworkModel("../cdf-01/cdf-10000.csv", inputFields, targetFields, learningRate,
		numLayers, epochs, numNeuronsPerLayer, activationFunction,
		distribution, Scale, pval)

	var lastLoss float64 = 0.08
	var i, count int = 0, 0
	var oldVLost float64 = 0.08
	for {

		newLoss, accuracy := nn.Train(network, 0)
		fmt.Println(" >>>>>>>> Train >>> Epoch  === ", i, " >>>>>>> New accuracy  === ", accuracy, " >>>>>>>> New Loss  === ", newLoss)

		if newLoss >= lastLoss {
			network.LearningRate = network.LearningRate / 10
			fmt.Println("Training loss increasing")
			log.Println("<<<<<<<<<< Decrease learning rate >>>>>>>>>>>>>", network.LearningRate)
			lastLoss = newLoss
		}

		if pval == true {
			vLoss, vAcc := nn.ValError(network)
			fmt.Println(" >>>>>>>> Valid >>> Epoch  === ", i, " >>>>>>> New accuracy  === ", vAcc, " >>>>>>>> New Loss  === ", vLoss)
			if vLoss >= oldVLost {
				oldVLost = vLoss
				//network.LearningRate = network.LearningRate / 10
				fmt.Println("Count ", count)
				count++
				if count > 1 {
					fmt.Println("Count >>> ", count)
					//break
				}
			}
		}

		// Early stopping
		if accuracy == 0.1 {
			nn.Test(network, pval)
			log.Println("<<<<<<<<<< Finished at Epoch = ", i, " learning rate = ", network.LearningRate, " >>>>>>>>>>>>>")
			nn.Save(network)
			break
		}
		i++
	}
}

/* Things left to do:
+ Test on 0.1% of data too
+ When validation goes up one time --> stop training
+ When accuracy requirement is met, stop training
+ Save and load weights and biases
- What causes loss to be stagnant?
+ use 1..n rather than "0001000.." for classification
+ train on combined cdf-30, 50, 100, 1000, 10000 into a single file?
+ try with pdf <> same results
+ try with kernel <> same results
*/
