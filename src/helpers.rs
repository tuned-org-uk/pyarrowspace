#![allow(non_local_definitions)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::PyReadonlyArray2;

use std::sync::atomic::{AtomicBool, Ordering};
static DEBUG: AtomicBool = AtomicBool::new(false);


#[pyfunction]
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, Ordering::Relaxed);
}

pub fn dbg_println(msg: impl AsRef<str>) {
    if DEBUG.load(Ordering::Relaxed) {
        eprintln!("[pyarrowspace] {}", msg.as_ref());
    }
}

// ------------ Helpers ------------
pub fn pyarray2_to_vecvec(items: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
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

pub fn parse_graph_params(
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