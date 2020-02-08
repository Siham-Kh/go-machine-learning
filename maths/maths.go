package maths

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

func GetRandomSlice(len int, v float64) (s []float64) {

	// why v?
	// The choice of the distribution is optional
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	s = make([]float64, len)
	for i := 0; i < len; i++ {
		s[i] = dist.Rand()
	}
	return
}

//
func InitWeights(in int, out int, v float64, activ int, dt bool) (s []float64) {

	len := in * out

	// Appearently the initialization is a uniform disturv

	switch activ {
	case 1:
		// He init works better for Relu activation
		// dt = false --> Uniform, if true --> Normal
		if dt == false {
			xavier := math.Sqrt(12 / float64(in+out))
			dist := distuv.Uniform{
				Min: (-1) * xavier,
				Max: xavier,
			}
			s = make([]float64, len)
			for i := 0; i < len; i++ {
				s[i] = dist.Rand()
			}
			return

		} else {
			// var dist distuv.Normal
			xavier := math.Sqrt(4 / float64(in+out))
			dist := distuv.Normal{
				Mu:    0,
				Sigma: xavier,
			}
			s = make([]float64, len)
			for i := 0; i < len; i++ {
				s[i] = dist.Rand()
			}
			return
		}

	default:
		// Xavier init works better for Sigmoid activation
		// dt = false --> Uniform, if true --> Normal
		if dt == false {
			xavier := math.Sqrt(6 / float64(in+out))
			dist := distuv.Uniform{
				Min: (-1) * xavier,
				Max: xavier,
			}
			s = make([]float64, len)
			for i := 0; i < len; i++ {
				s[i] = dist.Rand()
			}
			return

		} else {
			// var dist distuv.Normal
			xavier := math.Sqrt(2 / float64(in+out))
			dist := distuv.Normal{
				Mu:    0,
				Sigma: xavier,
			}
			s = make([]float64, len)
			for i := 0; i < len; i++ {
				s[i] = dist.Rand()
			}
		}
		return
	}
}

func Sum(nums ...float64) float64 {
	var sum float64 = 0
	for _, num := range nums {
		sum += num
	}
	return sum
}

func minSlice(s []float64) float64 {
	var min float64 = 0
	for i, e := range s {
		if i == 0 || e < min {
			min = e
		}
	}
	return min
}

func maxSlice(s []float64) float64 {
	var max float64 = 0
	for i, e := range s {
		if i == 0 || e > max {
			max = e
		}
	}
	return max
}

func NormSlice(s []float64) []float64 {

	min := minSlice(s)
	max := maxSlice(s)
	ss := make([]float64, len(s))

	if (max - min) == 0 {
		return nil
	} else {
		for i, e := range s {
			if e == 0 {
				ss[i] = 0
			}
			if e == min {
				ss[i] = 0
			}
			if (e != min) && (e != 0) {
				ss[i] = (e - min) / (max - min)
			}
		}

		return ss
	}
}

func mean(s []float64) float64 {

	var sum float64 = 0
	for _, e := range s {
		sum += e
	}

	return sum / float64(len(s))
}

func standardDeviation(s []float64) float64 {

	var v float64 = 0
	mean := mean(s)

	for _, e := range s {
		v += (e - mean) * (e - mean)
	}

	return math.Sqrt(v / float64(len(s)))
}

func StandarizeSlice(s []float64) []float64 {

	ss := make([]float64, len(s))

	xbar := mean(s)
	std := standardDeviation(s)

	for i, e := range s {

		ss[i] = (e - xbar) / std
	}

	return ss
}
