package nn

import (
	"Machine-learning/maths"

	"gonum.org/v1/gonum/mat"
)

type ValidSet struct {
	Inputs  *mat.Dense
	Outputs *mat.Dense
	Targets *mat.Dense
	Errors  *mat.Dense
}

// NewNetwork initializes a new neural network.
func NewValidSet(inputs *mat.Dense, Targets *mat.Dense) *ValidSet {

	ro, co := Targets.Dims()
	return &ValidSet{
		Inputs:  inputs,
		Targets: Targets,
		Outputs: mat.NewDense(ro, co, nil),
		Errors:  mat.NewDense(ro, co, nil),
	}
}

func ValError(net *Network) (float64, float64) {
	val := NewValidSet(net.VInputs, net.VTargets)
	ri, ci := val.Inputs.Dims()
	_, co := val.Targets.Dims()

	for input := 0; input < ri; input++ {

		vin := mat.NewDense(ci, 1, val.Inputs.RawRowView(input))
		vout := mat.NewDense(co, 1, nil)

		for index, layer := range net.Layers {

			tmp := mat.NewDense(layer.Neurons, 1, nil)
			tmp.Mul(layer.Weights, vin)
			tmp.Add(tmp, layer.Bias)

			switch net.Activation {
			case 1:
				//fmt.Println("Choosing Relu")
				tmp.Apply(maths.ReLu, tmp)
			default:
				//fmt.Println("Choosing Sigmoid")
				tmp.Apply(maths.Sigmoid, tmp)
			}

			if index == net.NHlayers-1 {
				vout.Copy(tmp)
			}

			vin.Reset()
			vin.ReuseAs(layer.Neurons, 1)
			vin.Copy(tmp)
		}
		// Final layer append it s results to validSet's Output
		val.Outputs.SetRow(input, vout.RawMatrix().Data)
	}
	val.Errors = maths.Subtract(val.Targets, val.Outputs).(*mat.Dense)

	// fmt.Printf(" Validation Target (from val) = %.9v\n", mat.Formatted(val.Targets))
	// fmt.Printf(" Validation Output  = %.9v\n", mat.Formatted(val.Outputs))
	// fmt.Printf(" Validation Error  = %v\n", mat.Formatted(val.Errors))

	acc := accuracy(val.Errors)
	dot := maths.Scale(0.5, maths.Multiply(val.Errors, val.Errors)).(*mat.Dense)
	sum := maths.AddMatrix(dot) / float64(ri)

	return sum, acc
}
