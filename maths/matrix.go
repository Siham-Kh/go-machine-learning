package maths

import (
	"log"

	"gonum.org/v1/gonum/mat"
)

func SumSlice(s []float64) float64 {
	var sum float64 = 0
	for _, val := range s {
		sum += val * val / 2
	}
	return sum
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

/* Check dimenssions later */
func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

/* add two matrices together */
func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

/* matrix multiplication m*n */
func Dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

/* row views of inputs */
func RowList(m *mat.Dense) []mat.Vector {
	r, _ := m.Dims()
	log.Println("r = ", r)
	var v []mat.Vector
	for i := 1; i <= r; i++ {
		v[i] = m.RowView(i)
	}
	return v
}

func AddMatrix(m *mat.Dense) float64 {

	r, c := m.Dims()
	var total float64 = 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			total += m.At(i, 0)
		}
	}
	return total
}
