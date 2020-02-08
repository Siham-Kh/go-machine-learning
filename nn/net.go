package nn

import (
	"Machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Neurons   int
	Weights   *mat.Dense
	Bias      *mat.Dense
	PreOutput *mat.Dense
	Output    *mat.Dense
	Error     *mat.Dense
	Records   int
	Delta     *mat.Dense
	wGrad     *mat.Dense
	bGrad     *mat.Dense
}

type Network struct {
	NHlayers     int
	Layers       []*Layer
	Inputs       *mat.Dense
	Target       *mat.Dense
	VInputs      *mat.Dense
	VTargets     *mat.Dense
	TestInputs   *mat.Dense
	TestTargets  *mat.Dense
	LearningRate float64
	Epochs       int
	Activation   int
	Distribution bool
}

// NewNetwork initializes a new neural network.
func NewNetwork(rate float64, nHlayers int, inputs *mat.Dense, target *mat.Dense, vinputs *mat.Dense, vtarget *mat.Dense, testInputs *mat.Dense, testTargets *mat.Dense, epochs int, activ int, distribution bool) *Network {
	return &Network{
		NHlayers:     nHlayers,
		Inputs:       inputs,
		Target:       target,
		VInputs:      vinputs,
		VTargets:     vtarget,
		TestInputs:   testInputs,
		TestTargets:  testTargets,
		LearningRate: rate,
		Epochs:       epochs,
		Activation:   activ,
		Distribution: distribution,
	}
}

// NewNetwork initializes a new neural network.
func (net *Network) NewLayer(numNeurons int, prevNeurons int, records int, b bool) *Layer {

	activ := net.Activation
	dist := net.Distribution

	wLayer := mat.NewDense(numNeurons, prevNeurons, maths.InitWeights(prevNeurons, numNeurons, float64(prevNeurons), activ, dist))
	wgrad := mat.NewDense(numNeurons, prevNeurons, nil)
	return &Layer{
		Neurons:   numNeurons,
		Weights:   wLayer,
		Bias:      bias(numNeurons, b),
		Records:   records,
		PreOutput: mat.NewDense(numNeurons, 1, nil),
		Output:    mat.NewDense(numNeurons, 1, nil),
		Error:     mat.NewDense(numNeurons, 1, nil),
		Delta:     mat.NewDense(numNeurons, 1, nil),
		bGrad:     mat.NewDense(numNeurons, 1, nil),
		wGrad:     wgrad,
	}
}

func bias(r int, bActivated bool) *mat.Dense {
	data := make([]float64, r)
	if bActivated {
		for i := range data {
			data[i] = float64(1)
		}
		return mat.NewDense(r, 1, data)
	} else {
		for i := range data {
			data[i] = float64(0)
		}
		return mat.NewDense(r, 1, data)
	}
}

func BuildNetworkModel(filename string, inputFields int, targetFields int,
	learningRate float64, numLayers int, epochs int, numNeuronsPerLayer []int, activ int, distribution bool, scale int, pval bool) *Network {

	/* 1 - Load Training data  Data */
	trainingInputs, trainingTargets, valInputs, valTargets, testInputs, testTargets := GetData(filename, inputFields, targetFields, scale, pval)

	/* 2 - Build Layers and model */
	trainingRecords, inputNodes := trainingInputs.Dims()
	network := NewNetwork(learningRate, numLayers, trainingInputs, trainingTargets, valInputs, valTargets, testInputs, testTargets, epochs, activ, distribution)

	for i := 0; i < numLayers; i++ {
		if i == 0 {
			firstLayer := network.NewLayer(numNeuronsPerLayer[i], inputNodes, trainingRecords, true)
			network.Layers = append(network.Layers, firstLayer)
		}
		if i == numLayers-1 {
			lastLayer := network.NewLayer(numNeuronsPerLayer[i], numNeuronsPerLayer[i-1], trainingRecords, false)
			network.Layers = append(network.Layers, lastLayer)
		}
		if (i != 0) && (i != numLayers-1) {
			midLayer := network.NewLayer(numNeuronsPerLayer[i], numNeuronsPerLayer[i-1], trainingRecords, true)
			network.Layers = append(network.Layers, midLayer)
		}
	}
	return network
}
