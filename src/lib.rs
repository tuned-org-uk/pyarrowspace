#![allow(non_local_definitions)]
use ::arrowspace::energymaps::{EnergyMaps, EnergyMapsBuilder};
use ::arrowspace::sampling::SamplerType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use smartcore::linalg::basic::arrays::Array;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ::arrowspace::builder::ArrowSpaceBuilder as RustBuilder;
use ::arrowspace::core::{ArrowItem, ArrowSpace};
use ::arrowspace::graph::GraphLaplacian;
use ::arrowspace::motives::Motives;
use ::arrowspace::subgraphs::*;

mod helpers;
mod energyparams;
mod sorted_index;
mod subgraphs;

use crate::helpers::*;
use crate::energyparams::*;
use crate::sorted_index::*;
use crate::subgraphs::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_python;

use std::path::PathBuf;

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

    /// taumode search using eigenmaps (use build)
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

        let query = ArrowItem::new(v, lambda_q);
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
            
            let query = ArrowItem::new(v, lambda_q);
            results.push(self.inner.search_lambda_aware(&query, k, tau));
        }
        
        Ok(results)
    }

    /// taumode hybrid search using eigenmaps (use build): cosine + energy
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

    /// spot_motives_eigen(cfg: dict) -> List[List[int]]
    /// Runs triangle-based motif spotting on this Laplacian (EigenMaps build).
    fn spot_motives_eigen(&self, gl: &PyGraphLaplacian, cfg: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<Vec<usize>>> {
        let rcfg = parse_motives_config(cfg)?;
        dbg_println(format!("spot_motives_eigen -- gl.inner.shape: {:?}", gl.inner.shape()));
        let motifs = gl.inner.spot_motives_eigen(&rcfg);
        Ok(motifs)
    }

    /// spot_motives_energy(gl: PyGraphLaplacian, cfg: dict) -> List[List[int]]
    /// Runs energy-aware motif spotting on the subcentroid graph and returns item-index motifs.
    fn spot_motives_energy(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<usize>>> {
        let rcfg = parse_motives_config(cfg)?;
        // Ensure mapping exists; if not, return empty or error
        if self.inner.centroid_map.is_none() {
            return Err(PyValueError::new_err(
                "centroid_map is None; build with EnergyMaps to use spot_motives_energy",
            ));
        }
        let motifs = gl.inner.spot_motives_energy(&self.inner, &rcfg);
        Ok(motifs)
    }

    /// spot_subg_motives(gl: GraphLaplacian, cfg: dict) -> List[dict]

    /// Runs energy-mode motif-based subgraph extraction and returns a list of
    /// subgraph dictionaries with:
    /// - "node_indices": List[int] centroid indices
    /// - "item_indices": Optional[List[int]] original item indices
    /// - "rayleigh": Optional[float] Rayleigh cohesion
    /// - "nnodes": int number of centroids
    /// - "nfeatures": int feature dimension
    fn spot_subg_motives(
        &self,
        gl: &PyGraphLaplacian,
        cfg: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        let rcfg = parse_subgraph_config(cfg)?;
        dbg_println(format!(
            "spot_subg_motives -- gl.shape={:?}, min_size={}, rayleigh_max={:?}",
            gl.inner.shape(),
            rcfg.min_size,
            rcfg.rayleigh_max
        ));

        let subgraphs = gl.inner.spot_subg_motives(&self.inner, &rcfg);

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
    ) -> PyResult<Vec<PyObject>> {
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
            Py::new(py, PyArrowSpace { inner: aspace })?,
            Py::new(py, PyGraphLaplacian { inner: gl })?,
        ))
    }

    /// Same as `build(...)` but save computations on parquet files
    #[staticmethod]
    pub fn build_and_store(
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

        let uuid = get_uid(py)?;
        let dataset_name = format!("dataset_{}", uuid);
        let cwd = get_python_cwd(py)?;
        
        let dir_path = cwd.join("storage");

        use std::fs;
        dbg_println(format!("Creating directory at: {:?}", dir_path.canonicalize().unwrap_or(dir_path.clone())));
        fs::create_dir_all(&dir_path).expect("Failed to create directory");
        dbg_println(format!("build: Storing in path {:?}", dir_path));
        
        if let Some((eps, k, topk, p, sigma)) = parse_graph_params(graph_params)? {
            builder = builder
                .with_lambda_graph(eps, k, topk, p, sigma)
                .with_dims_reduction(true, Some(eps))
                .with_seed(42)
                .with_sparsity_check(false)
                .with_persistence(dir_path, dataset_name);
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
                .with_extra_dims_reduction(true)
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
    m.add_class::<PyLambdasSortedIter>()?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    Ok(())
}
