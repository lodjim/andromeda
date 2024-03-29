# Andromeda: Fast Machine Learning and Deep Learning Algorithms in Go

Andromeda is a Golang library that focuses on implementing machine learning and deep learning algorithms with an emphasis on speed compared to traditional implementations in Python. Currently, the library supports the K-means clustering algorithm, and additional algorithms will be added in future releases.

## Installation

To use Andromeda in your Go project, you can simply use the `go get` command:

```bash
go get -u github.com/lodjim/andromeda
```

## Example Usage

Here is an example of using Andromeda to perform K-means clustering on a dataset:

```go

package main
import (
	"fmt"
	"os"
	"encoding/csv"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"github.com/yourusername/andromeda/Kmean"
)

func main() {
	// Open and read the CSV file
	file, err := os.Open("iris.csv")
	if err != nil {
		fmt.Println("Error opening CSV file:", err)
		return
	}
	defer file.Close()

	// Parse the CSV records
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV records:", err)
		return
	}

	// Prepare the data for K-means clustering
	var data []float64
	for _, record := range records[1:] {
		for _, value := range record[0 : len(record)-1] {
			floatValue, _ := strconv.ParseFloat(value, 64)
			data = append(data, floatValue)
		}
	}
	x := mat.NewDense(len(data)/4, 4, data)

	// Create a K-means model with 3 clusters
	model := &kmean.Kmean{3}

	// Fit the model to the data
	centroids, _ := model.Fit(x)

	// Predict the label and minimum distance for a new data point
	label, distanceMin := model.Predict([]float64{5.4, 3, 4.5, 1.5}, centroids)

	// Print the results
	fmt.Println("Predicted Label:", label)
	fmt.Println("Minimum Distance:", distanceMin)
}
```

## Contribution

Contributions to Andromeda are welcome! If you have ideas for new algorithms, optimizations, or improvements, feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/yourusername/andromeda).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.