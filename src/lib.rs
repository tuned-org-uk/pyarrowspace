#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ::arrowspace::energymaps::{EnergyMaps, EnergyMapsBuilder};

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
    // #[getter]
    // fn data<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
    //     // Assume ArrowSpace::data() -> &Vec<Vec<f64>>
    //     let rows: &Vec<Vec<f64>> = &self.inner.data_to_vec();
    //     // from_vec2 checks that all rows have the same length; map its error to a ValueError
    //     PyArray::from_vec2(py, rows)
    //         .map_err(|e| PyValueError::new_err(format!("non-rectangular data: {e}")))
    // }

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

    /// Energy-only search (no cosine similarity).
    /// 
    /// Parameters:
    ///   - item: query vector (must match nfeatures)
    ///   - gl: GraphLaplacian object (energy-based graph from build_energy)
    ///   - k: number of results to return
    ///   - w_lambda: weight for lambda proximity term (default: 1.0)
    ///   - w_dirichlet: weight for Rayleigh-Dirichlet term (default: 0.5)
    /// 
    /// Returns:
    ///   List of (index, score) tuples sorted by descending score
    fn search_energy(
        &self,
        py: Python<'_>,
        item: PyReadonlyArray1<f64>,
        gl: Py<PyGraphLaplacian>,
        k: usize,
        w_lambda: Option<f64>,
        w_dirichlet: Option<f64>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?.to_vec();
        if v.len() != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query length {} must match nfeatures {}",
                v.len(),
                self.inner.nfeatures
            )));
        }

        let gl_ref = gl.borrow(py);
        let graph_laplacian = &gl_ref.inner;

        let w_l = w_lambda.unwrap_or(1.0);
        let w_d = w_dirichlet.unwrap_or(0.5);

        dbg_println(format!(
            "search_energy: qlen={}, k={}, w_λ={:.2}, w_D={:.2}",
            v.len(), k, w_l, w_d
        ));

        Ok(self.inner.search_energy(&v, graph_laplacian, k, w_l, w_d))
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
                .with_dims_reduction(true, Some(eps))
                .with_seed(42)
                //.with_inline_sampling(None)
                //.with_spectral(true)
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


    /// Build ArrowSpace using energy-only pipeline (no cosine similarity).
    /// 
    /// This constructs a graph where edges are weighted purely by energy features:
    /// lambda proximity, dispersion, and Dirichlet smoothness. The pipeline removes
    /// all cosine similarity dependence from both construction and search.
    /// 
    /// Parameters:
    ///   - items: 2D numpy array (N × F) of input vectors
    ///   - energy_params: optional dict with keys:
    ///       - optical_tokens: int, target centroids after compression (default: None)
    ///       - trim_quantile: float, fraction to trim per bin (default: 0.1)
    ///       - eta: float, diffusion step size (default: 0.1)
    ///       - steps: int, diffusion iterations (default: 4)
    ///       - split_quantile: float, dispersion split threshold (default: 0.9)
    ///       - neighbor_k: int, neighborhood size (default: 8)
    ///       - split_tau: float, split offset magnitude (default: 0.15)
    ///       - w_lambda: float, lambda weight in distance (default: 1.0)
    ///       - w_disp: float, dispersion weight (default: 0.5)
    ///       - w_dirichlet: float, Dirichlet weight (default: 0.25)
    ///       - candidate_m: int, candidate pool size (default: 32)
    ///   - graph_params: optional dict with standard graph params (eps, k, topk, p, sigma)
    ///       Used for configuring builder defaults; most are overridden by energy pipeline
    /// 
    /// Returns:
    ///   Tuple of (PyArrowSpace, PyGraphLaplacian) with energy-based graph
    /// 
    /// Note:
    ///   Dimensionality reduction is automatically enabled (required for energy pipeline).
    ///   Build time is 2-3× slower than standard build() due to diffusion and splitting.
    #[staticmethod]
    pub fn build_energy(
        py: Python<'_>,
        items: PyReadonlyArray2<f64>,
        energy_params: Option<&PyDict>,
        graph_params: Option<&PyDict>,
    ) -> PyResult<(Py<PyArrowSpace>, Py<PyGraphLaplacian>)> {
        dbg_println("build_energy: Converting pyarray2 to Vec<Vec>");
        let rows = pyarray2_to_vecvec(items)?;
        
        let e_params = parse_energy_params(energy_params)?;
        dbg_println(format!(
            "build_energy: optical_tokens={:?}, w_λ={:.2}, w_G={:.2}, w_D={:.2}",
            e_params.optical_tokens, e_params.w_lambda, e_params.w_disp, e_params.w_dirichlet
        ));

        let mut builder = RustBuilder::new();
        
        // Apply graph params if provided (used for builder configuration)
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_seed(42)
                .with_sparsity_check(false);
        }
        
        // Enable dims reduction (required for energy pipeline)
        builder = builder.with_dims_reduction(true, Some(0.35));
        
        dbg_println("build_energy: Starting energy pipeline");
        let (aspace, gl_energy) = builder.build_energy(rows, e_params);
        
        dbg_println(format!(
            "build_energy complete: nitems={}, nfeatures={}, graph_nodes={}, lambdas_len={}",
            aspace.nitems,
            aspace.nfeatures,
            gl_energy.nnodes,
            aspace.lambdas().len()
        ));

        Ok((
            Py::new(py, PyArrowSpace { inner: aspace })?,
            Py::new(py, PyGraphLaplacian { inner: gl_energy })?,
        ))
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