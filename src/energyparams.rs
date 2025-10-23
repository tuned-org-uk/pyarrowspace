use ::arrowspace::energymaps::{EnergyMaps, EnergyMapsBuilder, EnergyParams};
use pyo3::types::PyDict;
use pyo3::prelude::*;

// Add helper to parse EnergyParams from Python dict
pub fn parse_energy_params(dict_opt: Option<&PyDict>) -> PyResult<EnergyParams> {
    let mut params = EnergyParams::default();
    
    if let Some(d) = dict_opt {
        if let Some(v) = d.get_item("optical_tokens") {
            params.optical_tokens = v.extract::<Option<usize>>()?;
        }
        if let Some(v) = d.get_item("trim_quantile") {
            params.trim_quantile = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("eta") {
            params.eta = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("steps") {
            params.steps = v.extract::<usize>()?;
        }
        if let Some(v) = d.get_item("split_quantile") {
            params.split_quantile = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("neighbor_k") {
            params.neighbor_k = v.extract::<usize>()?;
        }
        if let Some(v) = d.get_item("split_tau") {
            params.split_tau = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("w_lambda") {
            params.w_lambda = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("w_disp") {
            params.w_disp = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("w_dirichlet") {
            params.w_dirichlet = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("candidate_m") {
            params.candidate_m = v.extract::<usize>()?;
        }
    }
    
    Ok(params)
}