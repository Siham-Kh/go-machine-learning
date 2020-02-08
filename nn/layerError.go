package nn

import (
	"Machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

func (net *Network) error(inputIndex int) *mat.Dense {
	x := net.errorAndDelta(inputIndex)

	for i := 0; i < net.NHlayers-1; i++ {
		net.oneLayerDelta(inputIndex, net.Layers[i], net.Layers[i+1])
	}
	return x
}

func (net *Network) errorAndDelta(inputIndex int) *mat.Dense {
	_, c := net.Target.Dims()
	numlayers := net.NHlayers

	// Reset Error to null
	net.Layers[numlayers-1].Error.Reset()
	net.Layers[numlayers-1].Error.ReuseAs(net.Layers[numlayers-1].Neurons, 1)

	// Reset Delta to null
	net.Layers[numlayers-1].Delta.Reset()
	net.Layers[numlayers-1].Delta.ReuseAs(net.Layers[numlayers-1].Neurons, 1)

	// target of input i
	target := net.Target.RawRowView(inputIndex)
	target_i := mat.NewDense(c, 1, target)

	// output of input i
	finalLayer := net.Layers[numlayers-1]

	output_i := finalLayer.Output

	// error of input i
	outputError_i := maths.Subtract(target_i, output_i).(*mat.Dense)

	// fill in the container back
	net.Layers[numlayers-1].Error.Copy(outputError_i)

	var x mat.Matrix
	switch net.Activation {
	case 1:
		//fmt.Println("Choosing Relu")

		preoutputPrime := net.Layers[numlayers-1].PreOutput
		preoutputPrime.Apply(maths.ReluPrime, preoutputPrime)

		x = maths.Multiply(outputError_i, preoutputPrime)
	default:
		//fmt.Println("Choosing Sigmoid")
		x = maths.Multiply(outputError_i, maths.SigmoidPrime(net.Layers[numlayers-1].Output))
	}

	// Update delta with activation choice
	net.Layers[numlayers-1].Delta.Copy(x)

	return outputError_i
}

func (net *Network) oneLayerDelta(inputIndex int, currentLayer *Layer, nextLayer *Layer) {

	// Reset delta to null
	currentLayer.Delta.Reset()
	currentLayer.Delta.ReuseAs(currentLayer.Neurons, 1)
	var delta mat.Matrix

	switch net.Activation {
	case 1:
		//fmt.Println("Choosing Relu")

		preoutputPrime := currentLayer.PreOutput
		preoutputPrime.Apply(maths.ReluPrime, preoutputPrime)

		delta = maths.Multiply(maths.Dot(nextLayer.Weights.T(), nextLayer.Delta), preoutputPrime)

	default:
		//fmt.Println("Choosing Sigmoid")
		delta = maths.Multiply(maths.Dot(nextLayer.Weights.T(), nextLayer.Delta), maths.SigmoidPrime(currentLayer.Output))

	}
	currentLayer.Delta.Copy(delta)
}
