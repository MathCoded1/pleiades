mod linear_regression;

use linear_regression::LinearRegression;
use linear_regression::DataPoint;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example dataset: (x, y) pairs for the scatter plot
    let data = vec![
        DataPoint::new( 1.0, 2.0 ),
        DataPoint::new( 2.0,4.0 ),
        DataPoint::new(  3.0,5.0 ),
        DataPoint::new(4.0, 4.0 ),
        DataPoint::new( 5.0, 5.0 ),
        DataPoint::new( 5.0, 4.0 ),
        DataPoint::new( 1.5, 2.5 ),
        DataPoint::new(3.2, 4.8 ),
        DataPoint::new(4.8, 5.3 ),
    ];
    let mut lr = LinearRegression::load(&data);
    lr.calculate();
    // Calculate the linear regression coefficients
    let (slope, intercept) = lr.get_slope_intercept();

    // Print the results
    println!("{:?}",lr);

    // Plot the data with the linear regression line
    lr.plot_data_with_regression(
        slope,
        intercept,
        "Scatter Plot with Linear Regression",
        "X Axis",
        "Y Axis",
        "scatter_plot.png",
    )?;

    Ok(())
}