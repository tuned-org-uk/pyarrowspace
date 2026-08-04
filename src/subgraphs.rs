#![allow(non_local_definitions, dead_code)]
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ::arrowspace::analysis::subgraphs::{
    SubgraphConfig, CentroidGraphParams,
};

use crate::helpers::extract_count;

pub fn parse_subgraph_config(cfg: Option<&Bound<'_, PyDict>>) -> PyResult<SubgraphConfig> {
    // Lightweight parser: start from defaults and override if present.
    // Integer fields use `extract_count` so integral floats (e.g. `4.0`)
    // coerce instead of raising — consistent with `parse_graph_params`.
    let mut s = SubgraphConfig::default();

    if let Some(d) = cfg {
        if let Some(v) = d.get_item("min_size")? {
            s.min_size = extract_count(&v, "subgraph config", "min_size")?;
        }
        if let Some(v) = d.get_item("rayleigh_max")? {
            if v.is_none() {
                s.rayleigh_max = None
            } else {
                let val: f64 = v.extract()?;
                s.rayleigh_max = Some(val);
            }
        }
        if let Some(v) = d.get_item("top_l")? {
            s.motives.top_l = extract_count(&v, "subgraph config", "top_l")?;
        }
        if let Some(v) = d.get_item("min_triangles")? {
            s.motives.min_triangles = extract_count(&v, "subgraph config", "min_triangles")?;
        }
        if let Some(v) = d.get_item("min_clust")? {
            s.motives.min_clust = v.extract()?;
        }
        if let Some(v) = d.get_item("max_motif_size")? {
            s.motives.max_motif_size = extract_count(&v, "subgraph config", "max_motif_size")?;
        }
        if let Some(v) = d.get_item("max_sets")? {
            s.motives.max_sets = extract_count(&v, "subgraph config", "max_sets")?;
        }
        if let Some(v) = d.get_item("jaccard_dedup")? {
            s.motives.jaccard_dedup = v.extract()?;
        }
    }

    Ok(s)
}

pub fn parse_centroid_graph_params(cfg: Option<&Bound<'_, PyDict>>) -> PyResult<CentroidGraphParams> {
    let mut p = CentroidGraphParams::default();

    if let Some(d) = cfg {
        if let Some(v) = d.get_item("eps")? {
            p.eps = v.extract()?;
        }
        if let Some(v) = d.get_item("k")? {
            p.k = extract_count(&v, "centroid graph params", "k")?;
        }
        if let Some(v) = d.get_item("topk")? {
            p.topk = extract_count(&v, "centroid graph params", "topk")?;
        }
        if let Some(v) = d.get_item("p")? {
            p.p = v.extract()?;
        }
        if let Some(v) = d.get_item("sigma")? {
            let val: Option<f64> = v.extract()?;
            p.sigma = val;
        }
        if let Some(v) = d.get_item("normalise")? {
            p.normalise = v.extract()?;
        }
        if let Some(v) = d.get_item("sparsitycheck")? {
            p.sparsitycheck = v.extract()?;
        }
        if let Some(v) = d.get_item("min_centroids")? {
            p.min_centroids = extract_count(&v, "centroid graph params", "min_centroids")?;
        }
        if let Some(v) = d.get_item("max_depth")? {
            p.max_depth = extract_count(&v, "centroid graph params", "max_depth")?;
        }
        if let Some(v) = d.get_item("seed")? {
            let val: Option<u64> = v.extract()?;
            p.seed = val;
        }
    }

    Ok(p)
}
