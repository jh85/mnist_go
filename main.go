package main

import (
	"fmt"
	"os"
	"math"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

func frombuffer(fname string, offset int) ([]byte,error) {
	f,err := os.Open(fname)
	if err != nil {
		return nil,err
	}
	defer f.Close()
	finfo,err := f.Stat()
	if err != nil {
		return nil,err
	}
	file_size := int(finfo.Size())
	buf := make([]byte,file_size)
	num,err := f.Read(buf)
	if err != nil {
		return nil,err
	}
	if num != file_size {
		fmt.Println("size mismatch")
	}
	return buf[offset:],nil
}

func print_mat(M *mat.Dense) {
	out := mat.Formatted(M,mat.Prefix(""),mat.Squeeze())
	fmt.Println(out)
}

func gen_mat(row int, col int, is_random bool) *mat.Dense {
	if is_random == true {
		return mat.NewDense(row,col,nil)
	} else {
		v := make([]float64,row*col)
		for i := range v {
			v[i] = rand.NormFloat64()
		}
		return mat.NewDense(row,col,v)
	}
}

type Neuralnet struct {
	params map[string] *mat.Dense
	learning_rate float64
}

func NewNeuralnet() *Neuralnet {
	nn := new(Neuralnet)
	nn.params = make(map[string]*mat.Dense)
	nn.params["w1"] = gen_mat(28*28,100,true)
	nn.params["b1"] = gen_mat(    1,100,false)
	nn.params["w2"] = gen_mat(  100, 10,true)
	nn.params["b2"] = gen_mat(    1, 10,false)
	nn.learning_rate = 0.1
	return nn
}

func relu(x *mat.Dense) *mat.Dense {
	f := func(i,j int, v float64) float64 {
		if v < 0 {
			return 0
		} else {
			return v
		}
	}
	var tmp mat.Dense
	res := &tmp
	res.Apply(f,x)
	return res
}

func AddVecB(m *mat.Dense, v *mat.Dense) *mat.Dense {
	r,c := m.Dims()
	zero := mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			zero.Set(i,j, m.At(i,j)+v.At(0,j))
		}
	}
	return zero
}

func (nn *Neuralnet) predict(x *mat.Dense) *mat.Dense {
	batch_size,_ := x.Dims()

	w1 := nn.params["w1"]
	b1 := nn.params["b1"]
	w2 := nn.params["w2"]
	b2 := nn.params["b2"]

	// z1 = ReLU(x @ w1 + b1)
	tmp1 := mat.NewDense(batch_size,100,nil)
	tmp1.Mul(x,w1)
	tmp2 := AddVecB(tmp1,b1)
	z1 := relu(tmp2)

	// score = z1 @ w2 + b2
	tmp3 := mat.NewDense(batch_size,10,nil)
	tmp3.Mul(z1,w2)
	score := AddVecB(tmp3,b2)

	return score
}

func (nn *Neuralnet) loss(x *mat.Dense, t []int) float64 {
	y := nn.predict(x)
	r,c := y.Dims()
	res := mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		var max_val float64 = -10000
		for j := 0; j < c; j++ {
			if y.At(i,j) > max_val {
				max_val = y.At(i,j)
			}
		}

		var row_exp_total float64 = 0
		for j := 0; j < c; j++ {
			res.Set(i,j, math.Exp(y.At(i,j)-max_val))
			row_exp_total += res.At(i,j)
		}

		for j := 0; j < c; j++ {
			res.Set(i,j, res.At(i,j) / row_exp_total)
		}
	}

	var total float64 = 0
	for i := 0; i < r; i++ {
		total += -math.Log(res.At(i,t[i]) + 1e-7)
	}
	return total / float64(r)
}

func (nn *Neuralnet) accuracy(x *mat.Dense, t []int) float64 {
	y := nn.predict(x)
	r,c := y.Dims()
	count := 0
	for i := 0; i < r; i++ {
		var max_val float64 = -10000
		var max_loc int = 0
		for j := 0; j < c; j++ {
			if y.At(i,j) > max_val {
				max_val = y.At(i,j)
				max_loc = j
			}
		}
		if max_loc == t[i] {
			count += 1
		}
	}
	return float64(count) / float64(r)
}

func (nn *Neuralnet) gradient(x *mat.Dense, t []int) map[string]*mat.Dense {
	h := 1e-4
	lr := nn.learning_rate

	w1 := nn.params["w1"]
	b1 := nn.params["b1"]
	w2 := nn.params["w2"]
	b2 := nn.params["b2"]

	grads := make(map[string]*mat.Dense)
	
	// w1
	r,c := w1.Dims()
	res := mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			var orig_val float64 = w1.At(i,j)
			w1.Set(i,j,orig_val+h)
			fxh1 := nn.loss(x,t)
			w1.Set(i,j,orig_val-h)
			fxh2 := nn.loss(x,t)
			res.Set(i,j,lr * (fxh1 - fxh2) / (2.0*h))
			w1.Set(i,j,orig_val)
		}
	}
	grads["w1"] = res

	// b1
	r,c = b1.Dims()
	res = mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			var orig_val float64 = b1.At(i,j)
			b1.Set(i,j,orig_val+h)
			fxh1 := nn.loss(x,t)
			b1.Set(i,j,orig_val-h)
			fxh2 := nn.loss(x,t)
			res.Set(i,j,lr * (fxh1 - fxh2) / (2.0*h))
			b1.Set(i,j,orig_val)
		}
	}
	grads["b1"] = res

	// w2
	r,c = w2.Dims()
	res = mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			var orig_val float64 = w2.At(i,j)
			w2.Set(i,j,orig_val+h)
			fxh1 := nn.loss(x,t)
			w2.Set(i,j,orig_val-h)
			fxh2 := nn.loss(x,t)
			res.Set(i,j,lr * (fxh1 - fxh2) / (2.0*h))
			w2.Set(i,j,orig_val)
		}
	}
	grads["w2"] = res

	// b2
	r,c = b2.Dims()
	res = mat.NewDense(r,c,nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			var orig_val float64 = b2.At(i,j)
			b2.Set(i,j,orig_val+h)
			fxh1 := nn.loss(x,t)
			b2.Set(i,j,orig_val-h)
			fxh2 := nn.loss(x,t)
			res.Set(i,j,lr * (fxh1 - fxh2) / (2.0*h))
			b2.Set(i,j,orig_val)
		}
	}
	grads["b2"] = res
	return grads
}

func main() {
	fname_images := "train-images-idx3-ubyte"
	fname_labels := "train-labels-idx1-ubyte"
	orig_images,err := frombuffer(fname_images,16)
	if err != nil {
		fmt.Println(err)
		return
	}
	orig_labels,err := frombuffer(fname_labels,8)
	if err != nil {
		fmt.Println(err)
		return
	}

	data_size := 28*28
	train_size := len(orig_images)/data_size

	images := make([]float64,len(orig_images))
	for i := 0; i < len(orig_images); i++ {
		images[i] = float64(orig_images[i]) / float64(255)
	}

	labels := make([]int,len(orig_labels))
	for i := 0; i < len(orig_labels); i++ {
		labels[i] = int(orig_labels[i])
	}

	model := NewNeuralnet()

	batch_size := 5
	chunk := data_size*batch_size
	for i := 0; i < int(train_size/batch_size); i++ {
		x := mat.NewDense(batch_size,data_size,images[i*chunk:(i+1)*chunk])
		t := labels[i*batch_size:(i+1)*batch_size]
		loss := model.loss(x,t)
		acc := model.accuracy(x,t)
		fmt.Println(i,loss,acc)
		grads := model.gradient(x,t)

		w1 := model.params["w1"]
		b1 := model.params["b1"]
		w2 := model.params["w2"]
		b2 := model.params["b2"]

		dw1 := grads["w1"]
		db1 := grads["b1"]
		dw2 := grads["w2"]
		db2 := grads["b2"]

		w1.Sub(w1, dw1)
		b1.Sub(b1, db1)
		w2.Sub(w2, dw2)
		b2.Sub(b2, db2)
	}
	
	return
}
