//! Python bindings over `arrowspace::analysis::sequencing` (arrowspace-rs
//! >= 0.26.12): total orderings over the nodes of a feature-space graph
//! Laplacian.
//!
//! Two strategies are exposed as module-level functions:
//!
//! * [`sequence_by_lambda`] — spectral curriculum over per-node λ scores.
//! * [`sequence_by_graph`] — MST-chain seriation of the Laplacian graph.
//!
//! Both validate inputs eagerly and raise `ValueError` instead of letting the
//! Rust-side assertions surface as `PanicException` (#25 item 4 convention).

#![allow(non_local_definitions)]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::arrowspace::analysis::sequencing::{self, Sequence as RustSequence};

use crate::PyGraphLaplacian;

/// A total order over the nodes of a graph Laplacian.
///
/// * `order` — node indices in sequence order (int64 ndarray,
///   permutation of `0..nnodes`)
/// * `positions` — discrete coordinate per step, aligned with `order`
///   (float64 ndarray): λ score for `sequence_by_lambda`, DFS discovery
///   depth within its component for `sequence_by_graph`
/// * `components` — number of connected components traversed
#[pyclass(name = "Sequence")]
pub struct PySequence {
    inner: RustSequence,
}

#[pymethods]
impl PySequence {
    /// Node indices in sequence order.
    #[getter]
    fn order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(
            py,
            &self.inner.order.iter().map(|&i| i as i64).collect::<Vec<_>>(),
        )
    }

    /// Per-step coordinates aligned with `order`.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.positions)
    }

    /// Number of connected components traversed.
    #[getter]
    fn components(&self) -> usize {
        self.inner.components
    }

    fn __repr__(&self) -> String {
        format!(
            "Sequence(nodes={}, components={})",
            self.inner.order.len(),
            self.inner.components
        )
    }
}

fn to_py_sequence(seq: RustSequence) -> PySequence {
    PySequence { inner: seq }
}

/// Orders nodes by their per-node λ scores (spectral curriculum).
///
/// Ascending by default (low → high Rayleigh energy); pass `descending=True`
/// for the reverse. Ties break on ascending node index; output is fully
/// deterministic.
///
/// Accepts any 1-D float sequence (numpy array or Python list).
/// Raises `ValueError` when fewer than two scores are supplied.
#[pyfunction]
#[pyo3(signature = (lambdas, descending=false))]
pub fn sequence_by_lambda(
    lambdas: &Bound<'_, PyAny>,
    descending: bool,
) -> PyResult<PySequence> {
    // Zero-copy fast path for ndarrays; lists fall back to an owned copy.
    let arr = lambdas.extract::<PyReadonlyArray1<f64>>().ok();
    let lam_vec: Vec<f64>;
    let lam: &[f64] = match &arr {
        Some(a) => a.as_slice()?,
        None => {
            lam_vec = lambdas.extract()?;
            &lam_vec
        }
    };
    if lam.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "sequencing requires at least two items, got {}",
            lam.len()
        )));
    }
    Ok(to_py_sequence(sequencing::sequence_by_lambda(lam, descending)))
}

/// Serialises the nodes of a graph Laplacian by walking its minimum spanning
/// forest in DFS preorder from approximate diameter endpoints.
///
/// Each connected component yields one contiguous block; blocks are ordered
/// by descending size (ties on ascending smallest member index). Positions
/// report DFS discovery depth within each component's walk. Deterministic.
///
/// Raises `ValueError` when the Laplacian has fewer than two nodes.
#[pyfunction]
#[pyo3(signature = (gl))]
pub fn sequence_by_graph(py: Python, gl: &PyGraphLaplacian) -> PyResult<PySequence> {
    if gl.inner.nnodes < 2 {
        return Err(PyValueError::new_err(format!(
            "sequencing requires at least two nodes, got {}",
            gl.inner.nnodes
        )));
    }
    // Heavy combinatorial work runs GIL-free, consistent with other wrappers;
    // the n >= 2 precondition above rules out the only upstream panic path.
    let seq = py.detach(|| sequencing::sequence_by_graph(&gl.inner));
    Ok(to_py_sequence(seq))
}
