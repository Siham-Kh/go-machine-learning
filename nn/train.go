package nn

import (
	"fmt"
	"go-machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

func accuracy(m *mat.Dense) float64 {

	r, c := m.Dims()
	var acc float64 = 0
	dim := float64(r * c)

	for col := 0; col < c; col++ {
		for row := 0; row < r; row++ {
			if m.At(row, col) == 0 {
				acc++
			}
		}
	}
	return acc / dim
}

func Test(net *Network, pval bool) {

	var r, c, ct int
	if pval == true {
		r, c = net.TestInputs.Dims()
		_, ct = net.TestTargets.Dims()

		testOutput := mat.NewDense(r, ct, nil)
		for input := 0; input < r; input++ {
			net.Predict(input, c, 1)
			testOutput.SetRow(input, net.Layers[net.NHlayers-1].Output.RawMatrix().Data)
		}

		fmt.Printf(" Test Target = %.9v\n", mat.Formatted(net.TestTargets))
		fmt.Printf(" Test Output = %.9v\n", mat.Formatted(testOutput))
	}

	if pval == false {
		r, c = net.Inputs.Dims()
		_, ct = net.Target.Dims()

		Output := mat.NewDense(r, ct, nil)
		for input := 0; input < r; input++ {
			net.Predict(input, c, 0)
			Output.SetRow(input, net.Layers[net.NHlayers-1].Output.RawMatrix().Data)
		}

		fmt.Printf(" Target = %.9v\n", mat.Formatted(net.Target))
		fmt.Printf(" Output = %.9v\n", mat.Formatted(Output))
	}
}

func Train_SGD(net *Network) (float64, float64) {

	r, c := net.Inputs.Dims()
	var sum float64 = 0
	totatError := mat.NewDense(r, net.Layers[net.NHlayers-1].Neurons, nil)

	for input := 0; input < r; input++ {

		/* Feed forward prediction */
		net.Predict(input, c, 0)

		/* Error and Delta propagation */
		fError := net.error(input)
		totatError.SetRow(input, fError.RawMatrix().Data)
		dot := maths.Multiply(fError, fError).(*mat.Dense)
		sum += maths.AddMatrix(dot) / 2

		net.backPropagation_SGD(input, r, c)
	}
	acc := accuracy(totatError)
	x := sum / float64(r)

	return x, acc
}

func Train_GD(net *Network) (float64, float64) {

	r, c := net.Inputs.Dims()
	var sum float64 = 0
	totatError := mat.NewDense(r, net.Layers[net.NHlayers-1].Neurons, nil)

	/* Feed forward prediction */
	for input := 0; input < r; input++ {

		net.Predict(input, c, 0)
		fError := net.error(input)
		net.ComputeGradients(input, r, c)

		/* Calculate Loss function */
		dot := maths.Multiply(fError, fError).(*mat.Dense)
		sum += maths.AddMatrix(dot) / 2
		totatError.SetRow(input, fError.RawMatrix().Data)
	}
	net.backPropagation_GD()

	acc := accuracy(totatError)
	x := sum / float64(r)
	return x, acc
}

func Train_MGD(net *Network, batchSize int) (float64, float64) {

	r, c := net.Inputs.Dims()
	var sum float64 = 0
	totatError := mat.NewDense(r, net.Layers[net.NHlayers-1].Neurons, nil)
	numBatches := 0
	num := r % batchSize
	if num == 0 {
		numBatches = int(r / batchSize)

	} else {
		numBatches = int(r/batchSize) + 1
	}
	var m int = 0

	for k := 0; k < numBatches; k++ {

		if (k+1)*batchSize > r {
			m = r
		}
		if (k+1)*batchSize <= r {
			m = (k + 1) * batchSize
		}

		for input := k * batchSize; input < m; input++ {

			net.Predict(input, c, 0)
			fError := net.error(input)
			net.ComputeGradients(input, r, c)

			/* Calculate Loss function */
			dot := maths.Multiply(fError, fError).(*mat.Dense)
			sum += maths.AddMatrix(dot) / 2
			totatError.SetRow(input, fError.RawMatrix().Data)
		}
		net.backPropagation_GD()
	}
	acc := accuracy(totatError)
	x := sum / float64(r)
	return x, acc
}

func (net *Network) zeroGradient() {
	for _, layer := range net.Layers {

		layer.bGrad.Reset()
		layer.bGrad.ReuseAs(layer.Neurons, 1)

		r, c := layer.wGrad.Dims()
		layer.wGrad.Reset()
		layer.wGrad.ReuseAs(r, c)
	}
}

func Train(net *Network, batchSize int) (float64, float64) {

	var loss, acc float64

	switch batchSize {
	case 1: /* Stochastic batch gradient descent */
		loss, acc = Train_SGD(net)
	case 0: /* Batch gradient descent */
		loss, acc = Train_GD(net)
	default: /* Mini batch gradient descent */
		loss, acc = Train_MGD(net, batchSize)
	}
	return loss, acc
}
