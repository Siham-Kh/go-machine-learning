package nn

import (
	"go-machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

func (net *Network) ComputeGradients(input int, row int, col int) {

	numLayers := net.NHlayers

	for layerIndex := 0; layerIndex < numLayers; layerIndex++ {
		currentLayer := net.Layers[layerIndex]
		if layerIndex == 0 {
			previousLayerOutput := net.Inputs.RawRowView(input)
			previousLayerActivation := mat.NewDense(col, 1, previousLayerOutput)
			sumGradient(currentLayer, previousLayerActivation)
		} else {
			previousLayer := net.Layers[layerIndex-1]
			previousLayerActivation := previousLayer.Output
			sumGradient(currentLayer, previousLayerActivation)
		}
	}
}

func sumGradient(currentLayer *Layer, previousLayerActivation *mat.Dense) {

	wgrad := maths.Dot(currentLayer.Delta, previousLayerActivation.T()).(*mat.Dense)
	bgrad := currentLayer.Delta

	currentLayer.bGrad = maths.Add(currentLayer.bGrad, bgrad).(*mat.Dense)
	currentLayer.wGrad = maths.Add(currentLayer.wGrad, wgrad).(*mat.Dense)
}

// For each input record, update the weight of all layer --> all network model update
func (net *Network) backPropagation_GD() {
	for _, layer := range net.Layers {
		layer.Weights = maths.Add(layer.Weights, maths.Scale(net.LearningRate, layer.wGrad)).(*mat.Dense)
		layer.Bias = maths.Add(layer.Bias, maths.Scale(net.LearningRate, layer.bGrad)).(*mat.Dense)
	}
}
