#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use ::arrowspace::builder::ArrowSpaceBuilder as RustBuilder;
use ::arrowspace::core::{ArrowItem, ArrowSpace};
use ::arrowspace::graph::GraphLaplacian;

use std::sync::atomic::{AtomicBool, Ordering};
static DEBUG: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_python;

#[pyfunction]
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, Ordering::Relaxed);
}

fn dbg_println(msg: impl AsRef<str>) {
    if DEBUG.load(Ordering::Relaxed) {
        eprintln!("[pyarrowspace] {}", msg.as_ref());
    }
}

// ------------ Helpers ------------
fn pyarray2_to_vecvec(items: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
    let arr = items.as_array(); // zero-copy view via rust-numpy
    let (n, d) = (arr.nrows(), arr.ncols());
    if n == 0 || d == 0 {
        return Err(PyValueError::new_err("items must be non-empty 2D array"));
    }
    // Debug shape and quick stats
    dbg_println(format!("items shape: ({}, {})", n, d));
    let first5 = arr.row(0).iter().take(5).cloned().collect::<Vec<_>>();
    dbg_println(format!("items[0][:5]: {:?}", first5));
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in arr.iter() {
        if v.is_nan() {
            nan_cnt += 1;
        }
        if v.is_infinite() {
            inf_cnt += 1;
        }
    }
    dbg_println(format!("NaNs: {}, Infs: {}", nan_cnt, inf_cnt));
    Ok(arr.rows().into_iter().map(|r| r.to_vec()).collect())
}

fn parse_graph_params(
    dict_opt: Option<&PyDict>,
) -> PyResult<Option<(f64, usize, usize, f64, Option<f64>)>> {
    if let Some(d) = dict_opt {
        let eps = d
            .get_item("eps")
            .ok_or_else(|| PyValueError::new_err("graph_params['eps'] is required"))?
            .extract::<f64>()?;
        let k = d
            .get_item("k")
            .ok_or_else(|| PyValueError::new_err("graph_params['k'] is required"))?
            .extract::<usize>()?;
        let topk = d
            .get_item("topk")
            .ok_or_else(|| PyValueError::new_err("graph_params['topk'] is required"))?
            .extract::<usize>()?;
        let p = d
            .get_item("p")
            .ok_or_else(|| PyValueError::new_err("graph_params['p'] is required"))?
            .extract::<f64>()?;
        // If "sigma" is missing or None -> default to eps * 0.5
        let sigma = match d.get_item("sigma") {
            Some(v) => v.extract::<Option<f64>>()?.or(Some(eps * 0.5)),
            None => Some(eps * 0.5),
        };
        Ok(Some((eps, k, topk, p, sigma)))
    } else {
        Ok(None)
    }
}

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
            "GraphLaplacian cannot be constructed directly; use ArrowSpaceBuilder.build_with_graph",
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
    fn graph_params<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py); // This returns Bound<'py, PyDict>
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

    // inner.data exposes &Vec<Vec<f64>>
    // property: ArrowSpace.data -> numpy.ndarray (float64, 2D)
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        // Assume ArrowSpace::data() -> &Vec<Vec<f64>>
        let rows: &Vec<Vec<f64>> = &self.inner.data_to_vec();
        // from_vec2 checks that all rows have the same length; map its error to a ValueError
        PyArray::from_vec2(py, rows)
            .map_err(|e| PyValueError::new_err(format!("non-rectangular data: {e}")))
    }

    /// Return (features: np.ndarray[float64], lambda: float) for item at idx.
    fn get_item<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<(&'py PyArray1<f64>, f64)> {
        if idx >= self.inner.nitems {
            // choose one of ValueError or IndexError depending on API preference
            return Err(PyValueError::new_err(format!(
                "index {} out of range [0, {})",
                idx, self.inner.nitems
            )));
        }

        // Obtain the ArrowItem from the Rust space.
        let it: ArrowItem = self.inner.get_item(idx);

        // Example with getters; change as needed:
        let feats_vec = it.item.to_vec();
        let lam = it.lambda;

        // Materialize as NumPy array owned by Python
        let feats = PyArray1::from_vec(py, feats_vec);

        Ok((feats, lam))
    }

    fn lambdas<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.inner.lambdas().to_vec())
    }

    /// Search using lambda-aware similarity.
    /// Parameters:
    ///   - item: query vector (must match nfeatures)
    ///   - gl: GraphLaplacian object (required for computing synthetic lambda)
    ///   - tau: optional tau value for similarity weighting (default: 0.9)
    ///   - k: number of results to return (default: 3)
    fn search(
        &self,
        py: Python<'_>,
        item: PyReadonlyArray1<f64>,
        gl: Py<PyGraphLaplacian>,
        tau: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?.to_vec();
        if v.len() != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query length {} must match nfeatures {}",
                v.len(),
                self.inner.nfeatures
            )));
        }

        // Extract the inner GraphLaplacian from the Python wrapper
        let gl_ref = gl.borrow(py);
        let graph_laplacian = &gl_ref.inner;

        // Compute synthetic lambda for the query using prepare_query_item
        // This uses the ArrowSpace's configured TauMode
        let lambda_q = self.inner.prepare_query_item(&v, graph_laplacian);

        assert_ne!(
            lambda_q, 0.0,
            "The lambdas are zero, check the magnitude of items and eps."
        );

        dbg_println(format!(
            "search: qlen={}, lambda_q={:.6}",
            v.len(),
            lambda_q
        ));

        // Create query item with computed lambda
        let query = ArrowItem::new(v, lambda_q);
        let k = graph_laplacian.graph_params.topk;

        // Perform lambda-aware search
        // tau_weight controls feature similarity weight, (1-tau_weight) controls lambda proximity weight
        Ok(self.inner.search_lambda_aware(&query, k, tau))
    }

    /// Search using lambda-aware similarity.
    /// Parameters:
    ///   - item: query vector (must match nfeatures)
    ///   - gl: GraphLaplacian object (required for computing synthetic lambda)
    ///   - tau: optional tau value for similarity weighting (default: 0.9)
    ///   - k: number of results to return (default: 3)
    fn search_hybrid(
        &self,
        py: Python<'_>,
        item: PyReadonlyArray1<f64>,
        gl: Py<PyGraphLaplacian>,
        tau: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?.to_vec();
        if v.len() != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query length {} must match nfeatures {}",
                v.len(),
                self.inner.nfeatures
            )));
        }

        // Extract the inner GraphLaplacian from the Python wrapper
        let gl_ref = gl.borrow(py);
        let graph_laplacian = &gl_ref.inner;

        // Compute synthetic lambda for the query using prepare_query_item
        // This uses the ArrowSpace's configured TauMode
        let lambda_q = self.inner.prepare_query_item(&v, graph_laplacian);

        dbg_println(format!(
            "search: qlen={}, lambda_q={:.6}",
            v.len(),
            lambda_q
        ));

        // Create query item with computed lambda
        let query = ArrowItem::new(v, lambda_q);
        let k = graph_laplacian.graph_params.topk;

        // Perform lambda-aware search
        // tau_weight controls feature similarity weight, (1-tau_weight) controls lambda proximity weight
        Ok(self.inner.search_lambda_aware_hybrid(&query, k, tau))
    }
}

#[pyclass(name = "ArrowSpaceBuilder")]
pub struct PyArrowSpaceBuilder;

#[pymethods]
impl PyArrowSpaceBuilder {
    #[staticmethod]
    pub fn build(
        py: Python<'_>,
        graph_params: Option<&PyDict>,
        items: PyReadonlyArray2<f64>,
    ) -> (Py<PyArrowSpace>, Py<PyGraphLaplacian>) {
        dbg_println("Convert pyarray2 and Vec<Vec>");
        let rows = pyarray2_to_vecvec(items).unwrap();
        let mut builder = RustBuilder::new();
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params).unwrap() {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(true, None)
                .with_sparsity_check(false);
        }
        dbg_println("Building from rows");
        let (aspace, gl) = builder.build(rows);
        dbg_println(format!(
            "built ArrowSpace: nitems={}, nfeatures={}, lambdas_len={}",
            aspace.nitems,
            aspace.nfeatures,
            aspace.lambdas().len()
        ));
        (
            Py::new(py, PyArrowSpace { inner: aspace }).unwrap(),
            Py::new(py, PyGraphLaplacian { inner: gl }).unwrap(),
        )
    }
}

#[pymodule]
pub fn arrowspace(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArrowSpaceBuilder>()?;
    m.add_class::<PyArrowSpace>()?;
    m.add_class::<PyGraphLaplacian>()?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    Ok(())
}