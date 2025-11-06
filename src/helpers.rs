use pyo3::prelude::*;
use pyo3::{Bound, types::PyDict};
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

pub fn parse_graph_params(dict_opt: Option<&Bound<'_, PyDict>>) -> PyResult<Option<(f64, usize, usize, f64, Option<f64>)>> {
    let Some(d) = dict_opt else {
        return Ok(None);
    };

    let eps = d
        .get_item("eps")?
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(0.2);
    let k = d
        .get_item("k")?
        .and_then(|v| v.extract::<usize>().ok())
        .unwrap_or(8);
    let topk = d
        .get_item("topk")?
        .and_then(|v| v.extract::<usize>().ok())
        .unwrap_or(3);
    let p = d
        .get_item("p")?
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(2.0);
    let sigma = match d.get_item("sigma")? {
        Some(v) => v.extract::<f64>().ok(),
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
    -> PyResult<::arrowspace::motives::MotiveConfig>
{
    use ::arrowspace::motives::MotiveConfig as RCfg;
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
