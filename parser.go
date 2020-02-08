package main

/*	After unzipping distribution x
	select kernel_100
	extract y-axis
	convert to csv
	add type "1,0,0,0,0,0,0,0,0" at the end of each file (check labels below)
	save it such that each file represents an input


	Labels:
	-------
	uniform			1,0,0,0,0,0,0,0,0
	normal			0,1,0,0,0,0,0,0,0
	logistic		0,0,1,0,0,0,0,0,0
	expononetial	0,0,0,1,0,0,0,0,0
	double_exp		0,0,0,0,1,0,0,0,0
	half_normal		0,0,0,0,0,1,0,0,0
	half_logistic	0,0,0,0,0,0,1,0,0
	gumbel_min		0,0,0,0,0,0,0,1,0
	gumbel_max		0,0,0,0,0,0,0,0,1


*/
import (
	"bufio"
	"encoding/csv"
	"log"
	"os"
	"path/filepath"
	"strings"
)

func getLabel_01(s string) []string {
	var label []string
	switch s {
	case "uniform":
		label = []string{"1", "0", "0", "0", "0", "0", "0", "0", "0"}
	case "normal":
		label = []string{"0", "1", "0", "0", "0", "0", "0", "0", "0"}
	case "logistic":
		label = []string{"0", "0", "1", "0", "0", "0", "0", "0", "0"}
	case "exponential":
		label = []string{"0", "0", "0", "1", "0", "0", "0", "0", "0"}
	case "double_exponential":
		label = []string{"0", "0", "0", "0", "1", "0", "0", "0", "0"}
	case "half_normal":
		label = []string{"0", "0", "0", "0", "0", "1", "0", "0", "0"}
	case "half_logistic":
		label = []string{"0", "0", "0", "0", "0", "0", "1", "0", "0"}
	case "gumbel_min":
		label = []string{"0", "0", "0", "0", "0", "0", "0", "1", "0"}
	case "gumbel_max":
		label = []string{"0", "0", "0", "0", "0", "0", "0", "0", "1"}
	default:
		label = nil
	}
	return label
}

func getLabel_19(s string) string {
	var label string
	switch s {
	case "uniform":
		label = "1"
	case "normal":
		label = "2"
	case "logistic":
		label = "3"
	case "exponential":
		label = "4"
	case "double_exponential":
		label = "5"
	case "half_normal":
		label = "6"
	case "half_logistic":
		label = "7"
	case "gumbel_min":
		label = "8"
	case "gumbel_max":
		label = "9"
	default:
		label = ""
	}
	return label
}

func checkFileType(path string, sep string) bool {

	if strings.Contains(path, ".txt") != true {
		log.Println("Fiel name ", path, " answer = ", false)
		return false
	}
	s := strings.Split(path, ".")
	if len(s) <= 0 {
		log.Println("Fiel name ", path, " answer = ", false)
		return false
	} else {
		sp := "rand_" + sep + "_" // e.g. _30_
		n := s[len(s)-2]          // removing .txt
		if strings.Contains(n, sp) != true {
			log.Println("Fiel name ", path, " answer = ", false)
			return false
		} else {
			log.Println("Fiel name ", path, " answer = ", true)
			return true
		}
	}
}

func readFile(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

type coordinates struct {
	x string
	y string
}

func verifError(message string, err error) {
	if err != nil {
		log.Fatal(message, err)
	}
}

func findFiles(name string) ([]string, error) {
	var files []string
	root := "../" + name
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return files, nil
}

func main() {

	args := os.Args[1:]
	folder := args[0]
	sep := args[1]
	log.Println(folder, sep)
	//Get the name of a distribution
	var name string
	s := strings.Split(folder, "/")
	if len(s) > 0 {
		name = s[0]
	}

	log.Println(name)
	distLabel := getLabel_01(name)

	// List files in distribution folder
	files, err := findFiles(name)
	if err != nil {
		log.Fatalf("Error looping through the files: %s", err)
	}

	// Create destination file
	newFileName := "../kernel-01/kernel-" + sep + ".csv"
	distName, err := filepath.Abs(newFileName)
	verifError("Wron New File Name", err)

	endFile, err := os.OpenFile(distName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	verifError("Cannot create file", err)
	defer endFile.Close()

	// Loop through the files inside the distribution folder
	// Extract the ones with kernel_X
	// Exctract y-axis from each file
	// append them in the destination file
	// move to the next folder
	for _, file := range files {
		r := checkFileType(file, sep)
		if r == false {
			continue
		} else {
			lines, er := readFile(file)
			if er != nil {
				log.Fatalf("readLines: %s", err)
			}
			var cords []string
			for _, line := range lines {
				s := strings.Fields(line)
				if len(s) != 1 {
					continue
				}
				cords = append(cords, s[0])
			}
			// Label the input
			cords = append(cords, distLabel...)

			// Writing to file
			writer := csv.NewWriter(endFile)
			defer writer.Flush()
			err = writer.Write(cords)
			verifError("Cannot write to file", err)
		}
	}
}
