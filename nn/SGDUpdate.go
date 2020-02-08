package nn

import (
	"Machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

// For each input record, update the weight of all layer --> all network model update
func (net *Network) backPropagation_SGD(input int, row int, col int) {

	numLayers := net.NHlayers

	for layerIndex := 0; layerIndex < numLayers; layerIndex++ {

		currentLayer := net.Layers[layerIndex]

		if layerIndex == 0 {

			previousLayerOutput := net.Inputs.RawRowView(input)
			previousLayerActivation := mat.NewDense(col, 1, previousLayerOutput)

			net.singleLayerOneInputUpdate(currentLayer, previousLayerActivation, input)

		} else {

			previousLayer := net.Layers[layerIndex-1]
			previousLayerActivation := previousLayer.Output

			net.singleLayerOneInputUpdate(currentLayer, previousLayerActivation, input)

		}

	}
}

func (net *Network) singleLayerOneInputUpdate(currentLayer *Layer, previousLayerActivation *mat.Dense, index int) {

	// weight update
	w := maths.Add(currentLayer.Weights,
		maths.Scale(net.LearningRate, maths.Dot(currentLayer.Delta, previousLayerActivation.T()))).(*mat.Dense)

	rw, cw := currentLayer.Weights.Dims()
	currentLayer.Weights.Reset()
	currentLayer.Weights.ReuseAs(rw, cw)
	currentLayer.Weights.Copy(w)

	// bias update
	b := maths.Add(currentLayer.Bias, maths.Scale(net.LearningRate, currentLayer.Delta))

	currentLayer.Bias.Reset()
	currentLayer.Bias.ReuseAs(currentLayer.Neurons, 1)
	currentLayer.Bias.Copy(b)

}
