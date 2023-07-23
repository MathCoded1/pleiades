// Function to calculate the mean of a vector
fn mean(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    sum / (values.len() as f64)
}

// Function to calculate the covariance of two vectors
fn calc_covariance(x: &[f64], y: &[f64], x_mean: f64, y_mean: f64) -> f64 {
    let mut covariance = 0.0;
    for i in 0..x.len() {
        covariance += (x[i] - x_mean) * (y[i] - y_mean);
    }
    covariance / (x.len() as f64 - 1.0)
}

// Function to calculate the variance of a vector
fn calc_variance(values: &[f64], mean: f64) -> f64 {
    let mut variance = 0.0;
    for &value in values {
        variance += (value - mean).powi(2);
    }
    variance / (values.len() as f64 - 1.0)
}