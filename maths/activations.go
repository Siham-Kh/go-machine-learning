package maths

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// sigmoid implements the sigmoid function
// for use in activation functions.
func Sigmoid(r, c int, x float64) float64 {
	return 1.0 / (1 + math.Exp(-1*x))
}

func SigmoidPrime(m *mat.Dense) *mat.Dense {

	rows, _ := m.Dims()

	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)

	return Multiply(m, subtract(ones, m)).(*mat.Dense) // m * (1 - m)
}

// ReLu and ReLu prime
func ReLu(r, c int, x float64) float64 {
	return math.Max(0, x)
}

func ReluPrime(a, b int, y float64) float64 {

	var x float64 = 0

	if x < 0 {
		x = 0
	}
	if x > 0 {
		x = 1
	}
	if x == 0 {
		x = 0.000000000001
	}

	return x
}

func Tanh(r, c int, x float64) float64 {
	return 2*Sigmoid(r, c, 2*x) - 1
}
