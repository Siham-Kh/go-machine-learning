package nn

import (
	"go-machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

func (net *Network) Predict(inputIndex int, col int, inp int) {

	var in *mat.Dense
	// default is training
	if inp == 0 {
		// in is the container of a single input
		in = mat.NewDense(col, 1, net.Inputs.RawRowView(inputIndex))
	}
	// 1 is for testing
	if inp == 1 {
		// in is the container of a single input
		in = mat.NewDense(col, 1, net.TestInputs.RawRowView(inputIndex))
	}

	// Predict for all layers with for this single input
	layers := net.Layers
	for _, layer := range layers {

		hiddenActivation := predictForOneLayer(in, layer, net.Activation)

		in.Reset()
		r, c := hiddenActivation.Dims()
		in.ReuseAs(r, c)

		in.Copy(hiddenActivation)
	}
}

func predictForOneLayer(input_i *mat.Dense, l *Layer, activ int) *mat.Dense {

	l.PreOutput.Reset()
	l.PreOutput.ReuseAs(l.Neurons, 1)

	l.PreOutput.Mul(l.Weights, input_i)

	l.PreOutput.Add(l.PreOutput, l.Bias)

	l.Output.Reset()
	l.Output.ReuseAs(l.Neurons, 1)

	switch activ {
	case 1:
		//fmt.Println("Choosing Relu")
		l.Output.Apply(maths.ReLu, l.PreOutput)
	default:
		//fmt.Println("Choosing Sigmoid")
		l.Output.Apply(maths.Sigmoid, l.PreOutput)
	}

	return l.Output
}
