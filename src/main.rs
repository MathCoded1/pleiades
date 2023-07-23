mod linear_regression;
mod datapoint;
use datapoint::DataPoint;
use linear_regression::LinearRegression;
fn main() {
    // Example usage
    let data = vec![
        DataPoint { x: 1.0, y: 1.0 },
        DataPoint { x: 2.0, y: 2.0 },
        DataPoint { x: 3.0, y: 1.5 },
        DataPoint { x: 4.0, y: 3.0 },
        DataPoint { x: 5.0, y: 2.5 },
    ];

    let mut regression = LinearRegression::new(&data);
    regression.calculate().unwrap();

    println!("Coefficients: {:?}", regression.get_coefficients());
    println!("Intercept: {:?}", regression.get_intercept());
    println!("R-squared: {:?}", regression.get_goodness_of_fit());
    println!("P-values: {:?}", regression.get_p_values());
    println!("Confidence Intervals: {:?}", regression.get_confidence_intervals());
    println!("VIF Values: {:?}", regression.get_vif_values());
    println!("Durbin-Watson Statistic: {:?}", regression.get_durbin_watson_statistic());
    println!("Outliers: {:?}", regression.get_outliers());
    println!("Homoscedasticity: {:?}", regression.get_homoscedasticity());
    println!("Predictions: {:?}", regression.predict(&vec![6.0, 7.0, 8.0]));
    regression.plot("Scatter Plot with Linear Regression", "X-axis", "Y-axis", "scatter_plot.png").unwrap();

    let x_values: Vec<f64> = (0..=10).map(|i| (2.0 * i as f64 / 3.0) + 1.0).collect();
    let predictions = regression.predict(&x_values);
    for point in predictions {
        println!("x: {:.1}, y_pred: {:.2}", point.x, point.y);
    }
}