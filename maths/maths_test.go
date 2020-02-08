package maths

import "testing"

func TestReLu(t *testing.T) {
	x, y := 1, 1
	a := ReLu(x, y, 1)
	b := ReLu(x, y, 0)
	c := ReLu(x, y, -1)

	if a != 1 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", a, 1)
	}
	if b != 0 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", b, 0)
	}
	if c != 0 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", c, 0)
	}
}

func TestReluPrime(t *testing.T) {
	x, y := 1, 1
	a := ReluPrime(x, y, 1)
	b := ReluPrime(x, y, 0)
	c := ReluPrime(x, y, -1)

	if a != 1 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", a, 1)
	}
	if b != 0 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", b, 0)
	}
	if c != 0 {
		t.Errorf("Relu was incorrect, got: %f, want: %d.", c, 0)
	}
}
