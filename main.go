package main

import (
	kmean "andromeda/Kmean"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func main() {
	file, err := os.Open("iris.csv")
	if err != nil {
		fmt.Println("Error opening CSV file:", err)
		return
	}
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV records:", err)
		return
	}
	var data []float64
	for _, record := range records[1:] {
		for _, value := range record[0 : len(record)-1] {
			float_value, _ := strconv.ParseFloat(value, 64)
			data = append(data, float64(float_value))
		}
	}
	x := mat.NewDense(len(data)/4, 4, data)
	model := &kmean.Kmean{3}
	c, _ := model.Fit(x)
	label, distance_min := model.Predict([]float64{5.4, 3, 4.5, 1.5}, c)
	fmt.Println(label)
	fmt.Println(distance_min)

}
