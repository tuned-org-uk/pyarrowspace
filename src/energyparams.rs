use ::arrowspace::energymaps::EnergyParams;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn parse_energy_params(dict_opt: Option<&Bound<'_, PyDict>>) -> PyResult<EnergyParams> {
    let mut params = EnergyParams::default();
    
    if let Some(d) = dict_opt {
        if let Some(v) = d.get_item("optical_tokens")? {
            params.optical_tokens = v.extract::<usize>().ok();
        }
        if let Some(v) = d.get_item("trim_quantile")? {
            params.trim_quantile = v.extract().ok().unwrap_or(params.trim_quantile);
        }
        if let Some(v) = d.get_item("eta")? {
            params.eta = v.extract().ok().unwrap_or(params.eta);
        }
        if let Some(v) = d.get_item("steps")? {
            params.steps = v.extract().ok().unwrap_or(params.steps);
        }
        if let Some(v) = d.get_item("split_quantile")? {
            params.split_quantile = v.extract().ok().unwrap_or(params.split_quantile);
        }
        if let Some(v) = d.get_item("neighbor_k")? {
            params.neighbor_k = v.extract().ok().unwrap_or(params.neighbor_k);
        }
        if let Some(v) = d.get_item("split_tau")? {
            params.split_tau = v.extract().ok().unwrap_or(params.split_tau);
        }
        if let Some(v) = d.get_item("w_lambda")? {
            params.w_lambda = v.extract().ok().unwrap_or(params.w_lambda);
        }
        if let Some(v) = d.get_item("w_disp")? {
            params.w_disp = v.extract().ok().unwrap_or(params.w_disp);
        }
        if let Some(v) = d.get_item("w_dirichlet")? {
            params.w_dirichlet = v.extract().ok().unwrap_or(params.w_dirichlet);
        }
        if let Some(v) = d.get_item("candidate_m")? {
            params.candidate_m = v.extract().ok().unwrap_or(params.candidate_m);
        }
    }
    
    Ok(params)
}
