#![allow(non_local_definitions, dead_code)]
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyTypeError;

use ::arrowspace::analysis::subgraphs::{
    SubgraphConfig, CentroidGraphParams,
};

use crate::helpers::extract_count;

/// Reject unknown / misspelled keys in a cfg dict — same failure family as
/// `parse_graph_params` (#25 item 2, applied to the subgraph cfg parsers in
/// #35 finding 3): a typo'd knob previously fell back to the default silently.
fn reject_unknown_keys(d: &Bound<'_, PyDict>, known: &[&str], scope: &str) -> PyResult<()> {
    let mut unknown: Vec<String> = Vec::new();
    for key in d.keys() {
        let key_str: String = key.extract().unwrap_or_default();
        if !known.contains(&key_str.as_str()) {
            unknown.push(key_str);
        }
    }
    if !unknown.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "{}: unknown key(s) {} — accepted keys are {:?}",
            scope,
            unknown
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", "),
            known
        )));
    }
    Ok(())
}

pub fn parse_subgraph_config(cfg: Option<&Bound<'_, PyDict>>) -> PyResult<SubgraphConfig> {
    // Lightweight parser: start from defaults and override if present.
    // Integer fields use `extract_count` so integral floats (e.g. `4.0`)
    // coerce instead of raising — consistent with `parse_graph_params`.
    let mut s = SubgraphConfig::default();

    if let Some(d) = cfg {
        // Reject unknown keys up front — same contract as `parse_graph_params`
        // since 0.26.5 (#35 finding 3): a typo'd knob must not silently no-op.
        const KNOWN: &[&str] = &[
            "min_size",
            "rayleigh_max",
            "top_l",
            "min_triangles",
            "min_clust",
            "max_motif_size",
            "max_sets",
            "jaccard_dedup",
        ];
        reject_unknown_keys(d, KNOWN, "subgraph config")?;

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
        // Same unknown-key contract as `parse_graph_params` (#35 finding 3).
        const KNOWN: &[&str] = &[
            "eps",
            "k",
            "topk",
            "p",
            "sigma",
            "normalise",
            "sparsitycheck",
            "min_centroids",
            "max_depth",
            "seed",
        ];
        reject_unknown_keys(d, KNOWN, "centroid graph params")?;

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
