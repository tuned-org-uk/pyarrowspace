#![allow(non_local_definitions)]
use ::arrowspace::energymaps::{EnergyMaps, EnergyMapsBuilder};
use ::arrowspace::sampling::SamplerType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ::arrowspace::builder::ArrowSpaceBuilder as RustBuilder;
use ::arrowspace::core::{ArrowItem, ArrowSpace};
use ::arrowspace::graph::GraphLaplacian;

mod helpers;
mod energyparams;

use crate::helpers::*;
use crate::energyparams::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_python;

// ------------ Py wrappers ------------
#[pyclass(name = "GraphLaplacian")]
pub struct PyGraphLaplacian {
    inner: GraphLaplacian,
}

#[pymethods]
impl PyGraphLaplacian {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "GraphLaplacian cannot be constructed directly; use ArrowSpaceBuilder.build",
        ))
    }

    #[getter]
    fn nnodes(&self) -> usize {
        self.inner.nnodes
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn graph_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let params = &self.inner.graph_params;

        dict.set_item("eps", params.eps)?;
        dict.set_item("k", params.k)?;
        dict.set_item("topk", params.topk)?;
        dict.set_item("p", params.p)?;
        dict.set_item("sigma", params.sigma)?;

        Ok(dict)
    }
}

#[pyclass(name = "ArrowSpace")]
pub struct PyArrowSpace {
    inner: ArrowSpace,
}

#[pymethods]
impl PyArrowSpace {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "ArrowSpace cannot be constructed directly; use ArrowSpaceBuilder.build",
        ))
    }

    #[getter]
    fn nitems(&self) -> usize {
        self.inner.nitems
    }

    #[getter]
    fn nfeatures(&self) -> usize {
        self.inner.nfeatures
    }

    /// Return (features: np.ndarray[float64], lambda: float) for item at idx.
    fn get_item<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
        if idx >= self.inner.nitems {
            return Err(PyValueError::new_err(format!(
                "index {} out of range [0, {})",
                idx, self.inner.nitems
            )));
        }

        let it: ArrowItem = self.inner.get_item(idx);
        let feats_vec = it.item.to_vec();
        let lam = it.lambda;

        let feats = PyArray1::from_vec(py, feats_vec);

        Ok((feats, lam))
    }

    fn lambdas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.lambdas())
    }

    /// Get all data as 2D numpy array
    fn get_all_items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Collect all items into a flat vector
        let nitems = self.inner.nitems;
        let nfeatures = self.inner.nfeatures;
        
        let mut flat: Vec<f64> = Vec::with_capacity(nitems * nfeatures);
        for i in 0..nitems {
            let item = self.inner.get_item(i);
            flat.extend_from_slice(&item.item);
        }
        
        let arr = PyArray1::from_vec(py, flat);
        let shape = (nitems, nfeatures);
        
        Ok(arr.reshape(shape)?)
    }

    fn search(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;
        
        if v.len() != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query length {} must match nfeatures {}",
                v.len(),
                self.inner.nfeatures
            )));
        }

        let graph_laplacian = &gl.inner;
        let lambda_q = self.inner.prepare_query_item(v, graph_laplacian);

        if lambda_q == 0.0 {
            return Err(PyValueError::new_err(
                "Lambda is zero - check item magnitude and eps parameter"
            ));
        }

        dbg_println(format!("search: qlen={}, lambda_q={:.6}", v.len(), lambda_q));

        let query = ArrowItem::new(v.to_vec(), lambda_q);
        let k = graph_laplacian.graph_params.topk;

        Ok(self.inner.search_lambda_aware(&query, k, tau))
    }

    fn search_batch(
        &self,
        items: PyReadonlyArray2<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
    ) -> PyResult<Vec<Vec<(usize, f64)>>> {
        let arr = items.as_array();
        let (nqueries, nfeatures) = (arr.shape()[0], arr.shape()[1]);
        
        if nfeatures != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query features {} must match nfeatures {}",
                nfeatures, self.inner.nfeatures
            )));
        }

        let graph_laplacian = &gl.inner;
        let k = graph_laplacian.graph_params.topk;
        
        let mut results = Vec::with_capacity(nqueries);
        
        for i in 0..nqueries {
            let row = arr.row(i);
            let v = row.to_slice().unwrap();
            
            let lambda_q = self.inner.prepare_query_item(v, graph_laplacian);
            if lambda_q == 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Lambda is zero for query {} - check item magnitude and eps", i
                )));
            }
            
            let query = ArrowItem::new(v.to_vec(), lambda_q);
            results.push(self.inner.search_lambda_aware(&query, k, tau));
        }
        
        Ok(results)
    }

    fn search_hybrid(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;
        
        if v.len() != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query length {} must match nfeatures {}",
                v.len(),
                self.inner.nfeatures
            )));
        }

        let graph_laplacian = &gl.inner;
        let lambda_q = self.inner.prepare_query_item(v, graph_laplacian);

        dbg_println(format!("search_hybrid: qlen={}, lambda_q={:.6}", v.len(), lambda_q));

        let query = ArrowItem::new(v.to_vec(), lambda_q);
        let k = graph_laplacian.graph_params.topk;

        Ok(self.inner.search_lambda_aware_hybrid(&query, k, tau))
    }

    fn search_energy(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;

        let graph_laplacian = &gl.inner;

        dbg_println(format!(
            "search_energy: qlen={}, k={}",
            v.len(), k,
        ));

        Ok(self.inner.search_energy(v, graph_laplacian, k))
    }
}

#[pyclass(name = "ArrowSpaceBuilder")]
pub struct PyArrowSpaceBuilder;

#[pymethods]
impl PyArrowSpaceBuilder {
    #[staticmethod]
    pub fn build(
        py: Python<'_>,
        graph_params: Option<&Bound<'_, PyDict>>,
        items: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArrowSpace>, Py<PyGraphLaplacian>)> {
        dbg_println("build: Converting numpy array to internal format");
        
        let arr = items.as_array();
        let (nrows, ncols) = (arr.shape()[0], arr.shape()[1]);
        
        let rows: Vec<Vec<f64>> = if nrows > 1000 {
            use rayon::prelude::*;
            (0..nrows)
                .into_par_iter()
                .map(|i| arr.row(i).to_owned().to_vec())
                .collect()
        } else {
            (0..nrows)
                .map(|i| arr.row(i).to_owned().to_vec())
                .collect()
        };

        let mut builder = RustBuilder::new();
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(true, Some(eps))
                .with_seed(42)
                .with_sparsity_check(false);
        }

        dbg_println(format!("build: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl) = builder.build(rows);
        
        dbg_println(format!(
            "build complete: nitems={}, nfeatures={}, lambdas={}",
            aspace.nitems, aspace.nfeatures, aspace.lambdas().len()
        ));

        Ok((
            Py::new(py, PyArrowSpace { inner: aspace })?,
            Py::new(py, PyGraphLaplacian { inner: gl })?,
        ))
    }

    #[staticmethod]
    pub fn build_energy(
        py: Python<'_>,
        items: PyReadonlyArray2<f64>,
        energy_params: Option<&Bound<'_, PyDict>>,
        graph_params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Py<PyArrowSpace>, Py<PyGraphLaplacian>)> {
        dbg_println("build_energy: Converting numpy array");
        
        let arr = items.as_array();
        let (nrows, ncols) = (arr.shape()[0], arr.shape()[1]);
        
        let rows: Vec<Vec<f64>> = if nrows > 1000 {
            use rayon::prelude::*;
            (0..nrows)
                .into_par_iter()
                .map(|i| arr.row(i).to_owned().to_vec())
                .collect()
        } else {
            (0..nrows)
                .map(|i| arr.row(i).to_owned().to_vec())
                .collect()
        };
        
        let e_params = parse_energy_params(energy_params)?;
        dbg_println(format!(
            "build_energy: optical_tokens={:?}, w_λ={:.2}, w_G={:.2}, w_D={:.2}",
            e_params.optical_tokens, e_params.w_lambda, e_params.w_disp, e_params.w_dirichlet
        ));

        let mut builder = RustBuilder::new();
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(true, Some(eps))
                .with_seed(999)
                .with_inline_sampling(Some(SamplerType::Simple(0.6)))
                .with_spectral(false)
                .with_sparsity_check(false);
        }
        
        dbg_println(format!("build_energy: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl_energy) = py.allow_threads(|| {
            builder.build_energy(rows, e_params)
        });
        
        dbg_println(format!(
            "build_energy complete: nitems={}, nfeatures={}, graph_nodes={}, lambdas={}",
            aspace.nitems, aspace.nfeatures, gl_energy.nnodes, aspace.lambdas().len()
        ));

        Ok((
            Py::new(py, PyArrowSpace { inner: aspace })?,
            Py::new(py, PyGraphLaplacian { inner: gl_energy })?,
        ))
    }
}

#[pymodule]
pub fn arrowspace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArrowSpaceBuilder>()?;
    m.add_class::<PyArrowSpace>()?;
    m.add_class::<PyGraphLaplacian>()?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    Ok(())
}
