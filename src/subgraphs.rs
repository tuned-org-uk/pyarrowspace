#![allow(non_local_definitions, dead_code)]
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ::arrowspace::subgraphs::{
    SubgraphConfig, CentroidGraphParams,
};

pub fn parse_subgraph_config(cfg: Option<&Bound<'_, PyDict>>) -> PyResult<SubgraphConfig> {
    // Very lightweight parser: use defaults and override if present.
    let mut s = SubgraphConfig::default();

    if let Some(d) = cfg {
        if let Some(v) = d.get_item("min_size")? {
            s.min_size = v.extract()?;
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
            s.motives.top_l = v.extract()?;
        }
        if let Some(v) = d.get_item("min_triangles")? {
            s.motives.min_triangles = v.extract()?;
        }
        if let Some(v) = d.get_item("min_clust")? {
            s.motives.min_clust = v.extract()?;
        }
        if let Some(v) = d.get_item("max_motif_size")? {
            s.motives.max_motif_size = v.extract()?;
        }
        if let Some(v) = d.get_item("max_sets")? {
            s.motives.max_sets = v.extract()?;
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
            p.k = v.extract()?;
        }
        if let Some(v) = d.get_item("topk")? {
            p.topk = v.extract()?;
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
            p.min_centroids = v.extract()?;
        }
        if let Some(v) = d.get_item("max_depth")? {
            p.max_depth = v.extract()?;
        }
        if let Some(v) = d.get_item("seed")? {
            let val: Option<u64> = v.extract()?;
            p.seed = val;
        }
    }

    Ok(p)
}