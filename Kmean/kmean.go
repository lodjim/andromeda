package kmean

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Kmean struct {
	NumberOfCluster int
}

func calculateMean(lists [][]float64) []float64 {
	meanValue := make([]float64, 0)
	for i := 0; i < len(lists[1]); i++ {
		var meanNumber float64 = 0.0
		for _, list := range lists[1:] {
			meanNumber += list[i]
		}
		meanValue = append(meanValue, meanNumber/float64(len(lists)))
	}
	return meanValue
}

func (K Kmean) predictionNormalize(record []float64) []float64 {
	var min, max = record[0], record[0]
	for _, number := range record {
		if number < min {
			min = number
		}
		if number > max {
			max = number
		}
	}
	normRange := max - min

	for i, one_record := range record {
		record[i] = (one_record - min) / normRange
	}
	return record
}

func (K Kmean) minMaxNormalize(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	min, max := m.At(0, 0), m.At(0, 0)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m.At(i, j) < min {
				min = m.At(i, j)
			}
			if m.At(i, j) > max {
				max = m.At(i, j)
			}
		}
	}
	normRange := max - min
	X_Normalized := make([][]float64, r)
	for i := range X_Normalized {
		x_sample := make([]float64, c)
		for j := range x_sample {
			x_sample[j] = (m.At(i, j) - min) / normRange
		}
		X_Normalized[i] = x_sample
	}
	return X_Normalized
}

func (k Kmean) trainnigIsDone(old [][]float64, new [][]float64, threshold float64) bool {
	var distance float64
	for i := 0; i < len(old); i++ {
		distance += floats.Distance(old[i], new[i], 2)
	}
	prox := distance / float64(len(old))
	if prox < threshold {
		return true
	}
	return false
}

func (K Kmean) generateCentroid(m [][]float64) [][]float64 {
	c := len(m[0])
	listOfCentroid := make([][]float64, K.NumberOfCluster-1)
	for i := range listOfCentroid {
		centroidVector := make([]float64, c)
		for j := range centroidVector {
			rand.Seed(time.Now().UnixNano())
			centroidVector[j] = rand.Float64() - 0
		}
		listOfCentroid[i] = centroidVector
	}
	return listOfCentroid
}

func (K Kmean) Predict(record []float64, c [][]float64) (int, float64) {
	record = K.predictionNormalize(record)
	d_min := 100000.0
	label := 0
	var pred_label int
	for _, k := range c {

		distance := floats.Distance(record, k, 2)
		if distance < d_min {
			d_min = distance
			pred_label = label
		}
		label++
	}
	return pred_label, d_min
}

func (K Kmean) Fit(X *mat.Dense) ([][]float64, map[string]interface{}) {
	startTime := time.Now()
	K.NumberOfCluster++
	var clusterStructure map[string]interface{}
	X_Normalized := K.minMaxNormalize(X)
	var centroidVector [][]float64
	await_number := false
	for !await_number {
		clusterStructure = make(map[string]interface{})
		centroidVector = K.generateCentroid(X_Normalized)
		for i := range X_Normalized {
			d_min := 100000.0
			label := 0
			lb_cpt := 0
			for j := range centroidVector {
				distance := floats.Distance(X_Normalized[i], centroidVector[j], 2)
				if distance < d_min {
					d_min = distance
					label = lb_cpt
				}
				lb_cpt++
			}
			currentLabel := fmt.Sprintf("label%d", label)
			curVal, ok := clusterStructure[currentLabel]
			if !ok {
				curVal = [][]float64{{}}
			}
			sliceCurVal := curVal.([][]float64)
			sliceCurVal = append(sliceCurVal, X_Normalized[i])
			clusterStructure[currentLabel] = sliceCurVal
		}
		if len(clusterStructure) == (K.NumberOfCluster - 1) {
			await_number = true
		}
	}

	var new_centroidVector [][]float64
	for _, lbl := range clusterStructure {
		new_centroidVector = append(new_centroidVector, calculateMean(lbl.([][]float64)))
	}

	if K.trainnigIsDone(centroidVector, new_centroidVector, 0) {
		return new_centroidVector, clusterStructure
	} else {
		for K.trainnigIsDone(centroidVector, new_centroidVector, 0) {
			var centroidVector [][]float64
			centroidVector = new_centroidVector
			clusterStructure = make(map[string]interface{})
			for i := range X_Normalized {
				d_min := 100000.0
				label := 0
				lb_cpt := 0
				for j := range centroidVector {
					distance := floats.Distance(X_Normalized[i], centroidVector[j], 2)
					if distance < d_min {
						d_min = distance
						label = lb_cpt
					}
					lb_cpt++
				}
				currentLabel := fmt.Sprintf("label%d", label)
				curVal, ok := clusterStructure[currentLabel]
				if !ok {
					curVal = [][]float64{{}}
				}
				sliceCurVal := curVal.([][]float64)
				sliceCurVal = append(sliceCurVal, X_Normalized[i])
				clusterStructure[currentLabel] = sliceCurVal
			}

			var new_centroidVector [][]float64
			for _, lbl := range clusterStructure {
				new_centroidVector = append(new_centroidVector, calculateMean(lbl.([][]float64)))
			}
		}
	}
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	fmt.Printf("Execution Time: %s\n", elapsedTime)
	return new_centroidVector, clusterStructure
}
