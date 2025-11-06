use pyo3::prelude::*;

/// Python iterator yielding (lambda: float, idx: int)
#[pyclass]
pub struct PyLambdasSortedIter {
    data: Vec<(f64, usize)>,
    pos: usize,
}

#[pymethods]
impl PyLambdasSortedIter {
    #[new]
    pub fn new(data: Vec<(f64, usize)>) -> Self {
        Self { data, pos: 0 }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(f64, usize)> {
        if slf.pos >= slf.data.len() {
            return None;
        }
        let item = slf.data[slf.pos];
        slf.pos += 1;
        Some(item)
    }
}
