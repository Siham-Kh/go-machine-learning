package nn

import (
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func saveMatrix(m *mat.Dense, name string) {
	w, err := os.Create(name)
	defer w.Close()
	if err == nil {
		m.MarshalBinaryTo(w)
	}
}

func Save(net *Network) {

	for i, layer := range net.Layers {
		//save layer weight
		wname := "Layer_" + strconv.Itoa(i) + "_w.model"
		saveMatrix(layer.Weights, wname)

		// save layer bias
		bname := "Layer_" + strconv.Itoa(i) + "_b.model"
		saveMatrix(layer.Bias, bname)
	}
}

func loadMatrix(wname string, r int, c int) *mat.Dense {
	n := mat.NewDense(r, c, nil)
	w, err := os.Open(wname)
	defer w.Close()
	if err == nil {
		n.Reset()
		n.UnmarshalBinaryFrom(w)
	}
	return n
}

func Load(net *Network) {

	for i, layer := range net.Layers {
		//load layer weight
		wname := "Layer_" + strconv.Itoa(i) + "_w.model"
		r, c := layer.Weights.Dims()
		layer.Weights = loadMatrix(wname, r, c)

		// Load layer bias
		bname := "Layer_" + strconv.Itoa(i) + "_b.model"
		rb, cb := layer.Bias.Dims()
		layer.Bias = loadMatrix(bname, rb, cb)
	}
}
