#![allow(non_local_definitions)]
use ::arrowspace::maps::energymaps::{EnergyMaps, EnergyMapsBuilder};
use ::arrowspace::sampling::SamplerType;
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use smartcore::linalg::basic::arrays::Array;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ::arrowspace::builder::ArrowSpaceBuilder as RustBuilder;
use ::arrowspace::core::{ArrowItem, ArrowSpace};
use ::arrowspace::error::ArrowSpaceError;
use ::arrowspace::graph::GraphLaplacian;
use ::arrowspace::analysis::motives::Motives;
use ::arrowspace::analysis::subgraphs::*;

mod helpers;
mod energyparams;
mod sorted_index;
mod subgraphs;
mod sequencing;

use crate::helpers::*;
use crate::energyparams::*;
use crate::sorted_index::*;
use crate::subgraphs::*;
use crate::sequencing::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_python;

use std::path::PathBuf;

/// Map a fallible `ArrowSpaceError` from `try_*` variants to a typed,
/// catchable Python exception — never a `PanicException` (#25 item 4).
///
/// `DegenerateLambda` → `ValueError` (the guard `lib.rs` already wanted to
/// raise), `NonFiniteQuery` → `ValueError`, `DimensionMismatch` → `ValueError`,
/// `EnergyModeRequired` → `ValueError` (motives/subgraph APIs on EigenMaps
/// builds, #35 finding 1), `EigenModeRequired` → `ValueError` (item-space
/// eigen motifs on builds lacking the item→cluster bookkeeping, arrowspace
/// #165), `InvalidConfig` → `ValueError` (build_energy misconfiguration,
/// arrowspace #155).
fn map_arrow_error(e: ArrowSpaceError) -> PyErr {
    match e {
        ArrowSpaceError::DegenerateLambda { .. } => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::NonFiniteQuery => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::DimensionMismatch { .. } => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::EmptyItems => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::InvalidConfig { .. } => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::EnergyModeRequired { .. } => PyValueError::new_err(format!("{}", e)),
        ArrowSpaceError::EigenModeRequired { .. } => PyValueError::new_err(format!("{}", e)),
    }
}

fn get_python_cwd(py: Python) -> PyResult<PathBuf> {
    // Import the 'os' module
    let os = PyModule::import(py, "os")?;

    // Call the 'getcwd' function and extract the result as a Rust string
    let cwd_str: String = os.getattr("getcwd")?.call0()?.extract()?;

    // Convert the Rust string into a PathBuf for easier path manipulation
    Ok(PathBuf::from(cwd_str))
}

fn get_uid(py: Python) -> PyResult<String> {
    let uuid_mod = py.import("uuid")?;
    // Call uuid.uuid4() and convert to string
    let uid: String = uuid_mod.call_method0("uuid4")?.to_string();
    Ok(uid[..6].to_string())
}

use std::sync::Once;
static INIT: Once = Once::new();

/// Initialize logging for tests
pub fn init() {
    INIT.call_once(|| {
        pyo3_log::init();
    });
}

// ------------ Py wrappers ------------
#[pyclass(name = "GraphLaplacian")]
pub struct PyGraphLaplacian {
    inner: GraphLaplacian,
    storage_path: Option<String>,
    dataset_name: Option<String>,
}

#[pymethods]
impl PyGraphLaplacian {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "GraphLaplacian cannot be constructed directly; use ArrowSpaceBuilder().build",
        ))
    }

    #[getter]
    fn nnodes(&self) -> usize {
        self.inner.nnodes
    }

    /// Resolved persistence directory set by `build_and_store` / `with_persistence`
    /// builds (None for in-memory builds). See #25 item 5 / #17.
    #[getter]
    fn storage_path<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.storage_path {
            Some(s) => Ok(s.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    /// Resolved dataset name set by `build_and_store` / `with_persistence` builds
    /// (None for in-memory builds). Required by `load_arrowspace`.
    #[getter]
    fn dataset_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.dataset_name {
            Some(s) => Ok(s.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
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

    /// Export the sparse matrix in CSR format for NumPy/SciPy (f32).
    /// Returns (data: np.ndarray[f32], indices: np.ndarray[u64], indptr: np.ndarray[u64], shape: (int, int)).
    fn to_csr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        use pyo3::types::PyTuple; // Ensure this is imported

        let matrix = &self.inner.matrix;
        let (rows, cols) = matrix.shape();

        // Convert data to compatible types
        let indptr_vec: Vec<u64> = matrix.indptr().raw_storage().iter().map(|&x| x as u64).collect();
        let indices_vec: Vec<u64> = matrix.indices().iter().map(|&x| x as u64).collect();
        let data_vec: Vec<f32> = matrix.data().iter().map(|&x| x as f32).collect();

        // Create Bound<PyArray1> objects
        let py_data = PyArray1::from_vec(py, data_vec);
        let py_indices = PyArray1::from_vec(py, indices_vec);
        let py_indptr = PyArray1::from_vec(py, indptr_vec);
        
        // Create the shape tuple as a Bound<PyTuple>
        let py_shape = PyTuple::new(py, [rows, cols]).unwrap();

        // Combine everything into the final tuple.
        // We convert all items to Bound<PyAny> so they can be stored in the same array.
        let elements = [
            py_data.into_any(),
            py_indices.into_any(),
            py_indptr.into_any(),
            py_shape.into_any(),
        ];

        Ok(PyTuple::new(py, elements).unwrap())
    }

    /// Export as a dense NumPy array (f32) for direct PyTorch tensor conversion.
    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let matrix = &self.inner.matrix;
        let (rows, cols) = matrix.shape();
        
        // Initialize dense array
        let mut dense = vec![0.0f32; rows * cols];
        for (row_idx, row) in matrix.outer_iterator().enumerate() {
            for (col_idx, &value) in row.iter() {
                dense[row_idx * cols + col_idx] = value as f32;
            }
        }
        
        // Robust 1D -> Reshape pattern to avoid version conflicts with from_vec2
        let arr = PyArray1::from_vec(py, dense);
        Ok(arr.reshape((rows, cols))?)
    }
}

#[pyclass(name = "ArrowSpace")]
pub struct PyArrowSpace {
    inner: ArrowSpace,
    /// Resolved persistence addressing set by `build_and_store` / `with_persistence`
    /// builds, so the caller can reload its own artifacts (#25 item 5 / #17).
    storage_path: Option<String>,
    dataset_name: Option<String>,
}

#[pymethods]
impl PyArrowSpace {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "ArrowSpace cannot be constructed directly; use ArrowSpaceBuilder().build",
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

    /// Resolved persistence directory set by `build_and_store` / `with_persistence`
    /// builds (None for in-memory builds). Lets the caller reload its own
    /// artifacts without guessing — see #25 item 5 / #17.
    #[getter]
    fn storage_path<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.storage_path {
            Some(s) => Ok(s.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    /// Resolved dataset name set by `build_and_store` / `with_persistence` builds
    /// (None for in-memory builds). Required by `load_arrowspace`.
    #[getter]
    fn dataset_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.dataset_name {
            Some(s) => Ok(s.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    #[getter]
    fn nclusters(&self) -> usize {
        self.inner.n_clusters
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

    /// return computed lambdas
    fn lambdas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.lambdas())
    }

    /// Iterate over (lambda: float, idx: int) in ascending lambda; ties stable by id.
    pub fn lambdas_sorted(&self) -> Vec<(f64, usize)> {
        self.inner.lambdas_sorted.to_vec()
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

    /// taumode search using eigenmaps (use build).
    ///
    /// `k` overrides the number of retrieved neighbours (defaults to the index's
    /// `topk`). Passing `k=None` keeps the build-time `topk`. Separating the
    /// retrieval `k` from the construction `topk` avoids rebuilding the whole
    /// index for k-sweep experiments — see #25 item 3.
    ///
    /// Raises `ValueError` (not `PanicException`) for degenerate queries
    /// (lambda ~0), non-finite queries, or dimension mismatch — see #25 item 4.
    #[pyo3(signature = (item, gl, tau, k=None))]
    fn search(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
        k: Option<usize>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;

        let graph_laplacian = &gl.inner;
        // Fallible variant: surface a typed ValueError instead of letting the
        // upstream panic cross the FFI as an opaque PanicException (#25 item 4).
        let lambda_q = self
            .inner
            .try_prepare_query_item(v, graph_laplacian)
            .map_err(map_arrow_error)?;

        dbg_println(format!("search: qlen={}, lambda_q={:.6}", v.len(), lambda_q));

        let query = ArrowItem::new(v, lambda_q);
        let k = k.unwrap_or(graph_laplacian.graph_params.topk);

        // Fallible variant: the subcentroid path of `try_prepare_query_item`
        // can return `Ok(0.0)` (a stored subcentroid lambda of 0, no zero-guard
        // upstream), so the degeneracy surfaces here, not at prepare time. Use
        // `try_search_lambda_aware` and map to a catchable ValueError instead of
        // letting the infallible wrapper `.expect()` into a PanicException (#123).
        Ok(self
            .inner
            .try_search_lambda_aware(&query, k, tau)
            .map_err(map_arrow_error)?)
    }

    /// Batch taumode search. Degenerate rows (lambda ~0) yield `None` in that
    /// slot instead of aborting the whole batch — partial-failure tolerant, so
    /// whole-corpus work survives a single bad row. Valid rows are never
    /// discarded because of a later degenerate one. See #25 item 1.
    ///
    /// `k` overrides the retrieved-neighbour count (defaults to `topk`). See
    /// `search`. Non-finite or dimension-mismatched batches still raise (those
    /// are caller bugs affecting every row, not per-row degeneracy).
    ///
    /// # Energy / subcentroid indexes
    ///
    /// `search_batch` is **not implemented** for indexes built with
    /// `build_energy`: in that mode `try_prepare_query_item` maps the query to
    /// the nearest subcentroid and returns its *stored* lambda, which can be 0
    /// with no upstream guard (arrowspace-rs core.rs subcentroid branch). That
    /// makes per-row degeneracy undetectable at prepare time and would panic
    /// inside `search_lambda_aware`. Rather than ship a panic-prone batch path,
    /// we refuse energy-mode batches with `NotImplementedError` — use
    /// `search` (single) per row, or `search_hybrid` / `search_linear_sorted`,
    /// which tolerate the degenerate case. See #123.
    #[pyo3(signature = (items, gl, tau, k=None))]
    fn search_batch(
        &self,
        items: PyReadonlyArray2<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
        k: Option<usize>,
    ) -> PyResult<Vec<Option<Vec<(usize, f64)>>>> {
        if self.inner.sub_centroids.is_some() {
            return Err(PyNotImplementedError::new_err(
                "search_batch is not implemented for energy/subcentroid indexes \
                 (built with build_energy): the subcentroid path can return a \
                 degenerate lambda with no upstream guard, which would panic. \
                 Use search (single), search_hybrid, or search_linear_sorted. \
                 See #123.",
            ));
        }

        let arr = items.as_array();
        let (nqueries, nfeatures) = (arr.shape()[0], arr.shape()[1]);

        if nfeatures != self.inner.nfeatures {
            return Err(PyValueError::new_err(format!(
                "query features {} must match nfeatures {}",
                nfeatures, self.inner.nfeatures
            )));
        }

        let graph_laplacian = &gl.inner;
        let k = k.unwrap_or(graph_laplacian.graph_params.topk);

        let mut results: Vec<Option<Vec<(usize, f64)>>> = Vec::with_capacity(nqueries);

        for i in 0..nqueries {
            let row = arr.row(i);
            let v = row.to_slice().unwrap();

            match self.inner.try_prepare_query_item(v, graph_laplacian) {
                Ok(lambda_q) => {
                    let query = ArrowItem::new(v, lambda_q);
                    // `try_prepare_query_item` can return `Ok(0.0)` via the
                    // subcentroid path (no zero-guard upstream), so the
                    // degeneracy surfaces inside `try_search_lambda_aware`.
                    // Treat it as a per-row None instead of panicking the
                    // whole batch (#123).
                    match self.inner.try_search_lambda_aware(&query, k, tau) {
                        Ok(res) => results.push(Some(res)),
                        Err(ArrowSpaceError::DegenerateLambda { .. }) => {
                            results.push(None);
                        }
                        Err(e) => return Err(map_arrow_error(e)),
                    }
                }
                Err(ArrowSpaceError::DegenerateLambda { .. }) => {
                    // Skip this row, keep the rest of the batch.
                    results.push(None);
                }
                Err(e) => return Err(map_arrow_error(e)),
            }
        }

        Ok(results)
    }

    /// taumode hybrid search using eigenmaps (use build): cosine + energy.
    /// Raises `ValueError` for degenerate/non-finite/mismatched queries (#25 item 4).
    fn search_hybrid(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        tau: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;

        let graph_laplacian = &gl.inner;
        let lambda_q = self
            .inner
            .try_prepare_query_item(v, graph_laplacian)
            .map_err(map_arrow_error)?;

        dbg_println(format!("search_hybrid: qlen={}, lambda_q={:.6}", v.len(), lambda_q));

        let query = ArrowItem::new(v, lambda_q);
        let k = graph_laplacian.graph_params.topk;

        Ok(self.inner.search_lambda_aware_hybrid(&query, k, tau))
    }

    /// taumode energy search using energymaps (use build_energy)
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

    /// taumode search using sorted taumode (can be used with both builders)
    fn search_linear_sorted(
        &self,
        item: PyReadonlyArray1<f64>,
        gl: &PyGraphLaplacian,
        k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let v = item.as_slice()?;

        let graph_laplacian = &gl.inner;

        dbg_println(format!(
            "search_linear_sorted: qlen={}, k={}",
            v.len(), k,
        ));

        Ok(self.inner.search_linear_sorted(v, graph_laplacian, k))
    }

    /// spot_motives_eigen(gl: GraphLaplacian, cfg: dict) -> List[List[int]]
    /// Runs triangle-based motif spotting on this Laplacian (EigenMaps build).
    ///
    /// Node-space contract (arrowspace #165): the returned ids are the nodes
    /// of `gl.matrix` as built — on pipeline EigenMaps graphs that is the
    /// F×F bootstrap Laplacian, so the ids enumerate **feature dimensions,
    /// not items**, even though `gl.nnodes` reports the item count. For
    /// item-space motifs use `spot_motives_eigen_items`.
    #[allow(deprecated)] // deprecated in arrowspace 0.27.4; kept as the compatibility surface
    fn spot_motives_eigen(&self, gl: &PyGraphLaplacian, cfg: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<Vec<usize>>> {
        let rcfg = parse_motives_config(cfg)?;
        dbg_println(format!("spot_motives_eigen -- gl.inner.shape: {:?}", gl.inner.shape()));
        let motifs = gl.inner.spot_motives_eigen(&rcfg);
        Ok(motifs)
    }

    /// spot_motives_eigen_items(aspace, gl: GraphLaplacian, cfg: dict) -> List[List[int]]
    /// Item-space motif spotting on the EigenMaps track (arrowspace #165),
    /// mirroring `spot_motives_energy`:
    ///
    /// 1. Rebuilds the X×X Laplacian over cluster centroids from the index's
    ///    own rows and the item→cluster bookkeeping.
    /// 2. Detects motifs on the centroid graph.
    /// 3. Expands each centroid set to **item indices** and deduplicates.
    ///
    /// Requirements (enforced — raises `ValueError` mapped from
    /// `ArrowSpaceError::EigenModeRequired` instead of degrading):
    /// - `gl` must be an EigenMaps build (EnergyMaps graphs already have the
    ///   finer-grained `spot_motives_energy`)
    /// - every item carries an in-range cluster assignment
    /// - `n_clusters >= 2`
    ///
    /// Returned ids live in `0..n_items`; every motif is a union of whole
    /// clusters.
    fn spot_motives_eigen_items(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let rcfg = parse_motives_config(cfg)?;
        let motifs = gl
            .inner
            .try_spot_motives_eigen(&self.inner, &rcfg)
            .map_err(map_arrow_error)?;
        Ok(motifs)
    }

    /// spot_motives_energy(gl: GraphLaplacian, cfg: dict) -> List[List[int]]
    /// Runs energy-aware motif spotting on the subcentroid graph and returns item-index motifs.
    ///
    /// Requires an EnergyMaps build (`build_energy`); raises `ValueError`
    /// otherwise (mapped from `ArrowSpaceError::EnergyModeRequired`, #35).
    fn spot_motives_energy(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let rcfg = parse_motives_config(cfg)?;
        let motifs = gl
            .inner
            .try_spot_motives_energy(&self.inner, &rcfg)
            .map_err(map_arrow_error)?;
        Ok(motifs)
    }

    /// spot_subg_motives(gl: GraphLaplacian, cfg: dict) -> List[dict]
    /// Runs energy-mode motif-based subgraph extraction and returns a list of
    /// subgraph dictionaries with:
    /// - "node_indices": List[int] centroid indices
    /// - "item_indices": List[int] original item indices (energy builds only)
    /// - "rayleigh": Optional[float] Rayleigh cohesion — computed only when
    ///   cfg sets "rayleigh_max"; None otherwise (#35 finding 2)
    /// - "nnodes": int number of centroids
    /// - "nfeatures": int feature dimension
    ///
    /// Requires an EnergyMaps build (`build_energy`); raises `ValueError`
    /// otherwise (mapped from `ArrowSpaceError::EnergyModeRequired`, #35).
    fn spot_subg_motives(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let rcfg = parse_subgraph_config(cfg)?;
        dbg_println(format!(
            "spot_subg_motives -- gl.shape={:?}, min_size={}, rayleigh_max={:?}",
            gl.inner.shape(),
            rcfg.min_size,
            rcfg.rayleigh_max
        ));

        let subgraphs = gl
            .inner
            .try_spot_subg_motives(&self.inner, &rcfg)
            .map_err(map_arrow_error)?;

        Python::attach(|py| {
            let mut out = Vec::with_capacity(subgraphs.len());
            for sg in subgraphs {
                let dict = PyDict::new(py);

                dict.set_item("node_indices", sg.node_indices)?;
                if let Some(items) = sg.item_indices {
                    dict.set_item("item_indices", items)?;
                } else {
                    dict.set_item("item_indices", py.None())?;
                }
                if let Some(r) = sg.rayleigh {
                    dict.set_item("rayleigh", r)?;
                } else {
                    dict.set_item("rayleigh", py.None())?;
                }

                let (f_dim, x_dim) = sg.laplacian.init_data.shape();
                dict.set_item("nnodes", sg.laplacian.nnodes)?;
                dict.set_item("nfeatures", f_dim)?;
                dict.set_item("x_dim", x_dim)?;

                out.push(dict.into());
            }
            Ok(out)
        })
    }

    /// spot_subg_centroids(gl: GraphLaplacian, cfg: dict) -> List[dict]
    ///
    /// Builds a centroid hierarchy and returns all centroid-level subgraphs as a
    /// flat list of dictionaries with:
    /// - "level": int hierarchy depth
    /// - "node_indices": List[int] centroid indices (local to that level)
    /// - "root_indices": List[List[int]] original item indices per centroid
    /// - "nnodes": int centroid count at this level
    /// - "nfeatures": int feature dimension
    fn spot_subg_centroids(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let params = parse_centroid_graph_params(cfg)?;
        dbg_println(format!(
            "spot_subg_centroids -- gl.shape={:?}, max_depth={}, min_centroids={}",
            gl.inner.shape(),
            params.max_depth,
            params.min_centroids
        ));

        let hierarchy = gl.inner.build_centroid_hierarchy(&self.inner, params);

        Python::attach(|py| {
            let mut out = Vec::new();

            for (level_idx, level) in hierarchy.levels.iter().enumerate() {
                for node in level {
                    let dict = PyDict::new(py);

                    dict.set_item("level", level_idx)?;
                    dict.set_item("node_indices", &node.graph.node_indices)?;
                    dict.set_item("root_indices", &node.root_indices)?;

                    let (f_dim, x_dim) = node.graph.laplacian.init_data.shape();
                    dict.set_item("nnodes", node.graph.laplacian.nnodes)?;
                    dict.set_item("nfeatures", f_dim)?;
                    dict.set_item("x_dim", x_dim)?;

                    out.push(dict.into());
                }
            }

            Ok(out)
        })
    }
}

#[pyclass(name = "ArrowSpaceBuilder")]
pub struct PyArrowSpaceBuilder {
    pub(crate) inner: RustBuilder,
    /// Tracking copy of persistence addressing (dataset_name, storage_path) set
    /// via `with_persistence`, so `build()` can surface it on the returned
    /// objects. `RustBuilder.persistence` is `pub(crate)` in the upstream crate
    /// and therefore not readable here.
    persistence_info: Option<(String, String)>,
}

#[pymethods]
impl PyArrowSpaceBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustBuilder::new(),
            persistence_info: None,
        }
    }
    
    pub fn with_seed(mut slf: PyRefMut<Self>, seed: u64) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_seed(seed);
        slf
    }
    
    /// set dimensionality reduction and eps of the reduction
    pub fn with_dims_reduction(
        mut slf: PyRefMut<Self>,
        enabled: bool,
        eps: Option<f64>,
    ) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_dims_reduction(enabled, eps);
        slf
    }
    
    /// Set persistence addressing so `build(...)` writes Parquet files the caller
    /// can later reload via `load_arrowspace`. Without this, `build()` is
    /// in-memory only and `build_and_store()` picks a random unaddressable name.
    ///
    /// Exposed to Python for #17 (Option A): the adapter can now call
    /// `with_persistence(settings.index_store, slug)` so the write and reload
    /// paths agree on `dataset_name` and base directory, instead of the
    /// hardcoded CWD/`storage` + UUID that `build_and_store` used to generate.
    ///
    /// The resolved `(storage_path, dataset_name)` are surfaced as attributes on
    /// the returned `ArrowSpace` / `GraphLaplacian`.
    pub fn with_persistence(
        mut slf: PyRefMut<Self>,
        storage_path: String,
        dataset_name: String,
    ) -> PyRefMut<Self> {
        slf.persistence_info = Some((dataset_name.clone(), storage_path.clone()));
        slf.inner = slf
            .inner
            .clone()
            .with_persistence(PathBuf::from(&storage_path), dataset_name);
        slf
    }

    /// set sampling type and percentage
    pub fn with_sampling<'a>(mut slf: PyRefMut<'a, Self>, sampling: Option<&str>, value: Option<f64>) -> PyResult<PyRefMut<'a, Self>> {
        if sampling.is_some() || value.is_some() {
            assert!(sampling.is_some() && value.is_some(), "Should set smapling AND value")
        };
        if let Some(s) = sampling {
            let sampler = match s {
                "simple" => Some(SamplerType::Simple(value.unwrap())),
                "adaptive" => Some(SamplerType::DensityAdaptive(value.unwrap())),
                _ => return Err(PyValueError::new_err(format!("Unknown sampler: {}", s))),
            };
            slf.inner = slf.inner.clone().with_inline_sampling(sampler);
        }
        Ok(slf)
    }

    /// Set the maximum number of clusters manually.
    /// 
    /// If set, this overrides the automatic heuristic calculation.
    /// Use this to force a richer topology with more centroids.
    ///
    /// # Arguments
    /// * `max_clusters` - Target number of clusters (e.g., 150 for N=1150)
    ///
    /// # Example
    /// ```python
    /// builder = (ArrowSpaceBuilder()
    ///     .with_cluster_max_clusters(150)
    ///     .with_cluster_radius(0.85))
    /// ```
    pub fn with_cluster_max_clusters(
        mut slf: PyRefMut<Self>,
        max_clusters: usize,
    ) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_cluster_max_clusters(max_clusters);
        slf
    }

    /// Set the cluster radius (squared L2 threshold) manually.
    /// 
    /// Lower values create tighter, more numerous clusters.
    /// Default is 1.0. Typical range: [0.5, 2.0].
    ///
    /// # Arguments
    /// * `radius` - Squared L2 distance threshold for cluster creation
    ///
    /// # Example
    /// ```python
    /// builder = (ArrowSpaceBuilder()
    ///     .with_cluster_radius(0.85)  # Tighter clusters
    ///     .with_cluster_max_clusters(150))
    /// ```
    pub fn with_cluster_radius(
        mut slf: PyRefMut<Self>,
        radius: f64,
    ) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_cluster_radius(radius);
        slf
    }

    pub fn build(
        slf: PyRefMut<Self>,
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

        let mut builder = slf.inner.clone();
        let persist_info = slf.persistence_info.clone();
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_sparsity_check(false)
        }

        dbg_println(format!("build: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl) = py.detach(|| {
            let (aspace, gl) = builder.build(rows);
            
            dbg_println(format!(
                "build complete: nitems={}, nfeatures={}, lambdas={}",
                aspace.nitems, aspace.nfeatures, aspace.lambdas().len()
            ));

            (aspace, gl)
        });

        let (storage_path, dataset_name) = match persist_info {
            Some((name, path)) => (Some(path), Some(name)),
            None => (None, None),
        };
        Ok((
            Py::new(py, PyArrowSpace {
                inner: aspace,
                storage_path: storage_path.clone(),
                dataset_name: dataset_name.clone(),
            })?,
            Py::new(py, PyGraphLaplacian {
                inner: gl,
                storage_path,
                dataset_name,
            })?,
        ))
    }

    /// Same as `build(...)` but save computations on parquet files.
    ///
    /// `storage_path` / `dataset_name` make the write addressable so the caller
    /// can reload its own artifacts via `load_arrowspace` — previously the name
    /// was a random `dataset_<uuid>` surfaced only via `dbg_println`, and the
    /// path was a hardcoded CWD/`storage` (#25 item 5, #17). When omitted, the
    /// legacy behaviour (CWD/`storage`, random UUID) is preserved for backwards
    /// compatibility.
    ///
    /// The resolved `(storage_path, dataset_name)` are exposed as attributes on
    /// the returned `ArrowSpace` / `GraphLaplacian`.
    #[pyo3(signature = (graph_params, items, storage_path=None, dataset_name=None))]
    pub fn build_and_store(
        slf: PyRefMut<Self>,
        py: Python<'_>,
        graph_params: Option<&Bound<'_, PyDict>>,
        items: PyReadonlyArray2<f64>,
        storage_path: Option<String>,
        dataset_name: Option<String>,
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

        let dir_path = match &storage_path {
            Some(p) => PathBuf::from(p),
            None => {
                // Legacy default: CWD/storage
                let cwd = get_python_cwd(py)?;
                cwd.join("storage")
            }
        };
        let dataset_name = match dataset_name {
            Some(n) => n,
            None => format!("dataset_{}", get_uid(py)?),
        };

        use std::fs;
        dbg_println(format!("Creating directory at: {:?}", dir_path.canonicalize().unwrap_or(dir_path.clone())));
        // Raise a catchable OSError instead of panicking across the FFI when the
        // target directory is not writable (#25 item 4 minor).
        fs::create_dir_all(&dir_path).map_err(|e| {
            PyOSError::new_err(format!("Failed to create storage directory {:?}: {}", dir_path, e))
        })?;
        dbg_println(format!("build: Storing in path {:?} as {}", dir_path, dataset_name));

        let mut builder = slf.inner.clone();

        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_sparsity_check(false)
                .with_persistence(dir_path.clone(), dataset_name.clone());
        }

        dbg_println(format!("build: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl) = py.detach(|| {
            let (aspace, gl) = builder.build(rows);

            dbg_println(format!(
                "build complete: nitems={}, nfeatures={}, lambdas={}",
                aspace.nitems, aspace.nfeatures, aspace.lambdas().len()
            ));

            (aspace, gl)
        });

        let storage_path_str = dir_path.to_string_lossy().to_string();
        Ok((
            Py::new(py, PyArrowSpace {
                inner: aspace,
                storage_path: Some(storage_path_str.clone()),
                dataset_name: Some(dataset_name.clone()),
            })?,
            Py::new(py, PyGraphLaplacian {
                inner: gl,
                storage_path: Some(storage_path_str),
                dataset_name: Some(dataset_name),
            })?,
        ))
    }


    /// Like `build(...)` but no dim reduction
    pub fn build_full(
        slf: PyRefMut<Self>,
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

        let mut builder = slf.inner.clone();
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(false, None)
                .with_inline_sampling(None)
                .with_seed(42)
                .with_sparsity_check(false)
        }

        dbg_println(format!("build: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl) = py.detach(|| {
            let (aspace, gl) = builder.build(rows);
            
            dbg_println(format!(
                "build complete: nitems={}, nfeatures={}, lambdas={}",
                aspace.nitems, aspace.nfeatures, aspace.lambdas().len()
            ));

            (aspace, gl)
        });

        Ok((
            Py::new(py, PyArrowSpace { inner: aspace, storage_path: None, dataset_name: None })?,
            Py::new(py, PyGraphLaplacian { inner: gl, storage_path: None, dataset_name: None })?,
        ))
    }

    pub fn build_energy(
        slf: PyRefMut<Self>,
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

        let mut builder = slf.inner.clone();
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(true, Some(eps))
                .with_seed(999)
                .with_inline_sampling(Some(SamplerType::Simple(0.99)))
                .with_spectral(false)
                .with_sparsity_check(false);
        }
        
        dbg_println(format!("build_energy: Processing {} rows × {} cols", nrows, ncols));
        let (aspace, gl_energy) = py.detach(|| {
            let (aspace, gl_energy) = builder.build_energy(rows, e_params);
            
            dbg_println(format!(
                "build_energy complete: nitems={}, nfeatures={}, graph_nodes={}, lambdas={}",
                aspace.nitems, aspace.nfeatures, gl_energy.nnodes, aspace.lambdas().len()
            ));
            
            (aspace, gl_energy)
        });

        Ok((
            Py::new(py, PyArrowSpace { inner: aspace, storage_path: None, dataset_name: None })?,
            Py::new(py, PyGraphLaplacian { inner: gl_energy, storage_path: None, dataset_name: None })?,
        ))
    }
}

//
// load_arrowspace function for loading from storage
//

use ::arrowspace::graph::GraphParams;


/// Load ArrowSpace and GraphLaplacian from storage without recomputing.
/// 
/// # Arguments
/// * `storage_path` - Directory containing parquet files (e.g., "storage/")
/// * `dataset_name` - Prefix of the files (e.g., "dorothea_highdim")
/// * `graph_params` - Optional dict with graph parameters (eps, k, topk, p, sigma)
/// 
/// # Returns
/// Tuple of (ArrowSpace, GraphLaplacian)
/// 
/// # Example
/// ```python
/// aspace, gl = pyarrowspace.load_arrowspace(
///     storage_path="storage/",
///     dataset_name="dorothea_highdim",
///     graph_params={"eps": 0.5, "k": 10, "topk": 3, "p": 2.0}
/// )
/// ```
#[pyfunction]
pub fn load_arrowspace(
    py: Python,
    storage_path: String,
    dataset_name: String,
    graph_params: &Bound<'_, PyDict>,
    energy: bool,
) -> PyResult<(Py<PyArrowSpace>, Py<PyGraphLaplacian>)> {
    dbg_println(format!("Loading dataset '{}' from '{}'", dataset_name, storage_path));

    // Parse graph parameters
    let params_tuple = parse_graph_params(Some(graph_params))?;
    let g_params = if let Some((eps, k, topk, p, sigma)) = params_tuple {
        GraphParams { eps, k, topk, p, sigma, sparsity_check: false, normalise: false }
    } else {
        return Err(PyValueError::new_err(
            "Cannot parse GraphParams: graph_params dict is required for load_arrowspace"
        ));
    };

    dbg_println(format!("GraphParams {:?}", g_params));

    // Load ArrowSpace from storage
    dbg_println(format!("Loading aspace"));
    let aspace = py.detach(|| {
        ArrowSpace::new_from_storage(&storage_path, &dataset_name)
    }).map_err(|e| PyValueError::new_err(format!("Failed to load ArrowSpace: {}", e)))?;

    // Load GraphLaplacian from storage
    dbg_println(format!("Loading gl"));
    let gl = py.detach(|| {
        GraphLaplacian::new_from_storage(&storage_path, &dataset_name, g_params, energy)
    }).map_err(|e| PyValueError::new_err(format!("Failed to load GraphLaplacian: {}", e)))?;

    dbg_println(format!(
        "Loaded: {} items × {} features, {} GL nodes",
        aspace.nitems, aspace.nfeatures, gl.nnodes
    ));

    Ok((
        Py::new(py, PyArrowSpace {
            inner: aspace,
            storage_path: Some(storage_path.clone()),
            dataset_name: Some(dataset_name.clone()),
        })?,
        Py::new(py, PyGraphLaplacian {
            inner: gl,
            storage_path: Some(storage_path),
            dataset_name: Some(dataset_name),
        })?,
    ))
}

#[pymodule]
pub fn arrowspace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); 

    m.add_class::<PyArrowSpaceBuilder>()?;
    m.add_class::<PyArrowSpace>()?;
    m.add_class::<PyGraphLaplacian>()?;
    m.add_class::<PyLambdasSortedIter>()?;
    m.add_class::<PySequence>()?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    m.add_function(wrap_pyfunction!(load_arrowspace, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_by_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_by_graph, m)?)?;

    Ok(())
}
