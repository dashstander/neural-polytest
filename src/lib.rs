use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray2};
use rand::prelude::*;


struct GFPolynomial {
    p: u32,
    rng: StdRng,
}


impl GFPolynomial {
    fn new(p: u32, seed: u64) -> Self {
        Self {
            p,
            rng: StdRng::seed_from_u64(seed)
        }
    }

    fn random_poly(&mut self) -> Vec<u32> {
        (0..self.p).map(|_| self.rng.gen_range(0..self.p)).collect()
    }

    fn multiply(&self, p1: &[u32], p2: &[u32]) -> Vec<u32> {
        let n = self.p as usize;
        let mut result = vec![0; n];
        
        for i in 0..n {
            for j in 0..n {
                let pos = if i + j >= n {
                    // When degree would exceed n-1, reduce using x^n = x
                    ((i + j - n) % (n - 1)) + 1
                } else {
                    i + j
                };
                result[pos] = (result[pos] + (p1[i] * p2[j])) % self.p;
            }
        }    
        result
    }

    // Generate random batch with proper shape
    fn generate_batch(&mut self, batch_size: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut x_coeffs = Vec::with_capacity(batch_size);
        let mut y_coeffs = Vec::with_capacity(batch_size);
        let mut prod_coeffs = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let px = self.random_poly();
            let py = self.random_poly();
            let prod = self.multiply(&px, &py);
            
            x_coeffs.push(px);
            y_coeffs.push(py);
            prod_coeffs.push(prod);
        }

        (x_coeffs, y_coeffs, prod_coeffs)
    }

    // Generate all possible products
    fn generate_all(&self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let n = self.p as usize;
        let total = n.pow(n as u32);  // p^p combinations
        
        let mut x_coeffs = Vec::with_capacity(total * total);
        let mut y_coeffs = Vec::with_capacity(total * total);
        let mut prod_coeffs = Vec::with_capacity(total * total);

        // Generate all possible polynomials
        let mut all_polys: Vec<Vec<u32>> = Vec::with_capacity(total);
        
        fn generate_coeffs(p: u32, current: &mut Vec<u32>, all_polys: &mut Vec<Vec<u32>>) {
            if current.len() == p as usize {
                all_polys.push(current.clone());
                return;
            }
            for i in 0..p {
                current.push(i);
                generate_coeffs(p, current, all_polys);
                current.pop();
            }
        }
        
        generate_coeffs(self.p, &mut Vec::new(), &mut all_polys);

        // Generate all products
        for p1 in all_polys.iter() {
            for p2 in all_polys.iter() {
                let prod = self.multiply(p1, p2);
                x_coeffs.push(p1.clone());
                y_coeffs.push(p2.clone());
                prod_coeffs.push(prod);
            }
        }

        (x_coeffs, y_coeffs, prod_coeffs)
    }
}

#[pyclass]
struct PyGFPolynomial(GFPolynomial);

#[pymethods]
impl PyGFPolynomial {
    #[new]
    fn new(p: u32, seed: u64) -> Self {
        Self(GFPolynomial::new(p, seed))
    }

    fn generate_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize
    ) -> PyResult<(&'py PyArray2<u32>, &'py PyArray2<u32>, &'py PyArray2<u32>)> {
        let (x, y, prod) = self.0.generate_batch(batch_size);
        
        // Convert Shape errors to PyErrors
        let x_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            x.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let y_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            y.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let prod_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            prod.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            x_array.into_pyarray(py),
            y_array.into_pyarray(py),
            prod_array.into_pyarray(py)
        ))
    }

    fn generate_all<'py>(
        &self,
        py: Python<'py>
    ) -> PyResult<(&'py PyArray2<u32>, &'py PyArray2<u32>, &'py PyArray2<u32>)> {
        let (x, y, prod) = self.0.generate_all();
        let batch_size = x.len();
        
        let x_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            x.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let y_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            y.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let prod_array = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, self.0.p as usize),
            prod.into_iter().flatten().collect()
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            x_array.into_pyarray(py),
            y_array.into_pyarray(py),
            prod_array.into_pyarray(py)
        ))
    }
}

#[pymodule]
fn finite_fields(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGFPolynomial>()?;
    Ok(())
}