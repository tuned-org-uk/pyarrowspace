use pyo3::prelude::*;
use pyo3::{Bound, types::PyDict};
use pyo3::exceptions::PyTypeError;
use numpy::PyReadonlyArray2;

use std::sync::atomic::{AtomicBool, Ordering};
static DEBUG: AtomicBool = AtomicBool::new(false);


#[pyfunction]
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, Ordering::Relaxed);
}


pub fn dbg_println(s: impl AsRef<str>) {
    if cfg!(debug_assertions) {
        println!("{}", s.as_ref());
    }
}

/// Extract an integer count (`usize`) from a Python value, accepting either a
/// Python `int` or a float whose value is integral (e.g. `29.0` → `29`).
///
/// This exists because Python tuners (Optuna `trial.params`, JSON round-trips)
/// frequently deliver integer parameters as floats. Previously
/// `extract::<usize>()` rejected floats and the caller silently substituted a
/// default — see issue #23.
///
/// Behaviour:
///   * `int`           → returned as-is
///   * `29.0` (float)  → `29`, no warning (integral float is unambiguous)
///   * `29.5` (float)  → `PyTypeError` (non-integral counts are a caller bug)
///   * other types      → `PyTypeError`
fn extract_count(v: &Bound<'_, PyAny>, field: &str) -> PyResult<usize> {
    if let Ok(u) = v.extract::<usize>() {
        return Ok(u);
    }
    if let Ok(f) = v.extract::<f64>() {
        if f.is_finite() && f.fract() == 0.0 && f >= 0.0 {
            return Ok(f as usize);
        }
        return Err(PyTypeError::new_err(format!(
            "graph_params '{}': expected an integer count, got non-integral float {}",
            field, f
        )));
    }
    Err(PyTypeError::new_err(format!(
        "graph_params '{}': expected an integer or integral float, got {}",
        field,
        v.get_type().name()?
    )))
}

/// Parse a graph-params dict into the `(eps, k, topk, p, sigma)` tuple the
/// builder consumes, or `None` when no dict was supplied.
///
/// Defaults are applied **only for missing keys**. A key that is present but
/// unparseable raises `TypeError` — it is never silently dropped. Integral
/// floats (e.g. `29.0`) are accepted and coerced for `k`/`topk`; non-integral
/// floats (`29.5`) raise. `eps` and `p` are extracted as `f64` and accept
/// floats natively.
pub fn parse_graph_params(dict_opt: Option<&Bound<'_, PyDict>>) -> PyResult<Option<(f64, usize, usize, f64, Option<f64>)>> {
    let Some(d) = dict_opt else {
        return Ok(None);
    };

    let eps = match d.get_item("eps")? {
        Some(v) => v.extract::<f64>()
            .map_err(|e| PyTypeError::new_err(format!("graph_params 'eps': {}", e)))?,
        None => 0.2,
    };
    let k = match d.get_item("k")? {
        Some(v) => extract_count(&v, "k")?,
        None => 8,
    };
    let topk = match d.get_item("topk")? {
        Some(v) => extract_count(&v, "topk")?,
        None => 3,
    };
    let p = match d.get_item("p")? {
        Some(v) => v.extract::<f64>()
            .map_err(|e| PyTypeError::new_err(format!("graph_params 'p': {}", e)))?,
        None => 2.0,
    };
    let sigma = match d.get_item("sigma")? {
        Some(v) => match v.extract::<f64>() {
            Ok(f) => Some(f),
            Err(_) => None,
        },
        None => None,
    };

    Ok(Some((eps, k, topk, p, sigma)))
}

#[allow(dead_code)]
pub fn pyarray2_to_vecvec(arr: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
    let a = arr.as_array();
    let (nrows, _ncols) = (a.shape()[0], a.shape()[1]);
    
    let mut rows = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let row_view = a.row(i);
        rows.push(row_view.to_vec());
    }
    Ok(rows)
}

pub fn parse_motives_config(cfg: Option<&Bound<'_, PyDict>>)
    -> PyResult<::arrowspace::analysis::motives::MotiveConfig>
{
    use ::arrowspace::analysis::motives::MotiveConfig as RCfg;
    if let Some(d) = cfg {
        let top_l          = d.get_item("top_l")?.and_then(|v| v.extract::<usize>().ok()).unwrap_or(16);
        let min_triangles  = d.get_item("min_triangles")?.and_then(|v| v.extract::<usize>().ok()).unwrap_or(2);
        let min_clust      = d.get_item("min_clust")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.4);
        let max_motif_size = d.get_item("max_motif_size")?.and_then(|v| v.extract::<usize>().ok()).unwrap_or(32);
        let max_sets       = d.get_item("max_sets")?.and_then(|v| v.extract::<usize>().ok()).unwrap_or(256);
        let jaccard_dedup  = d.get_item("jaccard_dedup")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.8);
        Ok(RCfg {
            top_l,
            min_triangles,
            min_clust,
            max_motif_size,
            max_sets,
            jaccard_dedup,
            ..Default::default()
        })
    } else {
        Ok(RCfg::default())
    }
}
