package vector

import (
	"gonum.org/v1/gonum/mat"
)

// floats32To64 converts []float32 to []float64
func floats32To64(x []float32) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = float64(v)
	}
	return y
}

// floats64To32 converts []float64 to []float32
func floats64To32(x []float64) []float32 {
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = float32(v)
	}
	return y
}

// reduceDimensions reduces the dimensionality of a vector using SVD
func reduceDimensions(vec []float32, targetDim int) []float32 {
	if len(vec) <= targetDim {
		return vec
	}

	// Convert input vector to matrix format (1×N)
	m := mat.NewDense(1, len(vec), floats32To64(vec))

	// Perform SVD
	var svd mat.SVD
	ok := svd.Factorize(m, mat.SVDThin)
	if !ok {
		// If SVD fails, return truncated vector
		return vec[:targetDim]
	}

	// Project to target dimensions using U matrix
	u := mat.NewDense(1, len(vec), nil)
	svd.UTo(u)
	proj := mat.NewDense(1, targetDim, nil)
	proj.Mul(m, u.Slice(0, len(vec), 0, targetDim))

	return floats64To32(proj.RawRowView(0))
} 