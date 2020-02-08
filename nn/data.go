package nn

import (
	"go-machine-learning/maths"
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func GetData(fileName string, inputFields int, targetFields int, scale int, pval bool) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// Open the dataset file.
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = inputFields + targetFields

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, inputFields*len(rawCSVData))
	targetData := make([]float64, targetFields*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var targetIndex int

	// Sequentially move the rows into a slice of floats.
	for _, record := range rawCSVData {

		// Loop over the float columns.

		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i >= inputFields {
				targetData[targetIndex] = parsedVal
				targetIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// total untreated data from file
	inputs := mat.NewDense(len(rawCSVData), inputFields, inputsData)
	targets := mat.NewDense(len(rawCSVData), targetFields, targetData)

	// total scaled data: choice of scaling technique: 0 for normalization and 1 for standarization
	min := mat.NewDense(len(rawCSVData), inputFields, inputsData)
	//mout := mat.NewDense(len(rawCSVData), targetFields, targetData)

	switch scale {
	case 0:
		// normalization if 0
		min = NormalizeByColumn(inputs)
		//mout = NormalizeByColumn(targets)
	default:
		// standarization if 1
		min = StandarizeByColumn(inputs)
		//mout = StandarizeByColumn(targets)
	}

	if pval == true {
		return selectValidationSet(min, targets, 0.1)
	} else {
		return min, targets, nil, nil, nil, nil
	}
}

func selectValidationSet(in, out *mat.Dense, val float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {

	ri, ci := in.Dims()
	_, co := out.Dims()
	num := int(val*float64(ri)) + 1

	tin := mat.NewDense(ri-2*num, ci, nil)
	tout := mat.NewDense(ri-2*num, co, nil)

	vin := mat.NewDense(num, ci, nil)
	vout := mat.NewDense(num, co, nil)

	tstin := mat.NewDense(num, ci, nil)
	tstout := mat.NewDense(num, co, nil)

	rand.Seed(time.Now().Unix())

	s := make([]int, ri)
	for i := 0; i < ri; i++ {
		s[i] = i
	}
	rand.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
	newS := s[:num]
	midS := s[num : 2*num]
	restS := s[2*num:]

	for j := 0; j < num; j++ {
		index := newS[j]
		in.SetRow(j, in.RawRowView(index))
		vout.SetRow(j, out.RawRowView(index))
	}

	for j := 0; j < num; j++ {
		index := midS[j]
		tstin.SetRow(j, in.RawRowView(index))
		tstout.SetRow(j, out.RawRowView(index))
	}

	for k := 0; k < ri-2*num; k++ {
		index := restS[k]
		tin.SetRow(k, in.RawRowView(index))
		tout.SetRow(k, out.RawRowView(index))
	}

	return tin, tout, vin, vout, tstin, tstout
}

func NormalizeByColumn(m *mat.Dense) *mat.Dense {

	r, c := m.Dims()
	var colSlice []float64
	newMat := mat.NewDense(r, c, nil)

	for i := 0; i < c; i++ {
		col := m.ColView(i)
		/* Normalize a column by column */

		// from Column to slice
		for j := 0; j < r; j++ {
			val := col.AtVec(j)
			colSlice = append(colSlice, val)
		}

		// Normalize slice
		normCol := maths.NormSlice(colSlice)
		if normCol == nil {
			normCol = colSlice
		}

		// Add the normalized column to the new matrix
		newMat.SetCol(i, normCol)

		// Clear sli	ces for a new column
		colSlice = colSlice[:0]
		normCol = normCol[:0]
	}

	return newMat
}

func StandarizeByColumn(m *mat.Dense) *mat.Dense {

	r, c := m.Dims()
	var colSlice []float64
	newMat := mat.NewDense(r, c, nil)

	for i := 0; i < c; i++ {
		col := m.ColView(i)
		/* Normalize a column by column */

		// from Column to slice
		for j := 0; j < r; j++ {
			val := col.AtVec(j)
			colSlice = append(colSlice, val)
		}

		// Normalize slice
		stdCol := maths.StandarizeSlice(colSlice)

		if stdCol == nil {
			stdCol = colSlice
		}

		// Add the normalized column to the new matrix
		newMat.SetCol(i, stdCol)

		// Clear slices for a new column
		colSlice = colSlice[:0]
		stdCol = stdCol[:0]
	}

	return newMat
}
