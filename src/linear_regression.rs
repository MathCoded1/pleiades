use std::error::Error;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::style::full_palette::{RED, BLUE, WHITE};
use plotters::drawing::IntoDrawingArea;
use plotters::element::{Circle, EmptyElement};
use plotters::series::{LineSeries, PointSeries};
use crate::datapoint;
use datapoint::DataPoint;
use statrs::distribution::StudentsT;
use statrs::distribution::ContinuousCDF;

#[derive(Debug)]
pub(crate) struct LinearRegression {
    data: Box<Vec<DataPoint>>,
    coefficient: Option<f64>,
    intercept: Option<f64>,
    goodness_of_fit: Option<f64>,
    residuals: Option<Vec<f64>>,
    p_value_slope: Option<f64>,
    p_value_intercept: Option<f64>,
    confidence_level: Option<f64>,
    confidence_intervals: Option<(f64, f64)>,
    predictions: Option<f64>,
    multicollinearity: Option<f64>,
    outliers: Option<f64>,
    homoscedasticity: Option<f64>,
}

impl LinearRegression {
    pub(crate) fn new(data: &Vec<DataPoint>) -> Self {
        LinearRegression {
            data: Box::new(data.to_vec()),
            coefficient: None,
            intercept: None,
            goodness_of_fit: None,
            residuals: None,
            p_value_slope: None,
            p_value_intercept: None,
            confidence_level: None,
            confidence_intervals: None,
            predictions: None,
            multicollinearity: None,
            outliers: None,
            homoscedasticity: None,
        }
    }

    pub(crate) fn calculate(&mut self) -> Result<(), Box<dyn Error>> {
        self.calc_slope_intercept()?;
        self.calc_goodness_of_fit();
        self.calc_residuals();
        self.p_values_and_significance();
        self.calc_confidence_level_and_intervals();

        Ok(())
    }

    pub(crate) fn get_slope_intercept(&self) -> (f64, f64) {
        (self.coefficient.expect("Not Calculated"), self.intercept.expect("Not Calculated"))
    }

    // Function to calculate the linear regression coefficients
    fn calc_slope_intercept(&mut self) -> Result<(), Box<dyn Error>> {
        let n = self.data.len() as f64;
        let x_sum: f64 = self.data.iter().map(|p| p.x).sum();
        let y_sum: f64 = self.data.iter().map(|p| p.y).sum();
        let xy_sum: f64 = self.data.iter().map(|p| p.x * p.y).sum();
        let x_squared_sum: f64 = self.data.iter().map(|p| p.x * p.x).sum();

        let denominator = n * x_squared_sum - x_sum * x_sum;

        if denominator == 0.0 {
            return Err("Denominator is zero".into());
        }

        self.coefficient = Some((n * xy_sum - x_sum * y_sum) / denominator);
        self.intercept = Some((y_sum - self.coefficient.unwrap() * x_sum) / n);

        Ok(())
    }

    // Function to plot the data and the linear regression line
    pub(crate) fn plot(
        &self,
        plot_title: &str,
        x_label: &str,
        y_label: &str,
        output_file: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Create a plotter area
        let output_file = &format!("./images/plots/{}", output_file);
        let root = BitMapBackend::new(output_file, (1600, 1200)).into_drawing_area();
        root.fill(&WHITE)?;

        // Define the plot area
        let mut chart = ChartBuilder::on(&root)
            .caption(plot_title, ("sans-serif", 20))
            .margin(5)
            .set_all_label_area_size(40)
            .build_cartesian_2d(0.0..6.0, 0.0..6.0)?;

        // Draw the scatter plot
        chart.configure_mesh().draw()?;
        chart.draw_series(PointSeries::of_element(
            self.data.iter().map(|p| (p.x, p.y)),
            5,
            &BLUE,
            &|c, s, st| {
                EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
            },
        ))?;

        // Draw the linear regression line
        chart.draw_series(LineSeries::new(
            vec![
                (0.0, self.intercept.expect("Not Calculated")),
                (5.0, self.coefficient.expect("Not Calculated") * 5.0
                    + self.intercept.expect("Not Calculated")),
            ],
            &RED,
        ))?;

        Ok(())
    }

    // Function to calculate the goodness of fit (R-squared)
    fn calc_goodness_of_fit(&mut self) {
        let y_mean: f64 = self.data.iter().map(|p| p.y).sum::<f64>() / self.data.len() as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for point in self.data.iter() {
            let y_pred = self.coefficient.expect("Not Calculated") * point.x
                + self.intercept.expect("Not Calculated");
            let y_actual = point.y;

            ss_res += (y_actual - y_pred).powi(2);
            ss_tot += (y_actual - y_mean).powi(2);
        }

        self.goodness_of_fit = Some(1.0 - ss_res / ss_tot);
    }

    // Function to get the R-squared value
    pub(crate) fn r_squared(&self) -> f64 {
        self.goodness_of_fit.expect("Not Calculated")
    }

    pub(crate) fn calc_residuals(&mut self) {
        if let (Some(coefficient), Some(intercept)) = (self.coefficient, self.intercept) {
            let residuals: Vec<f64> = self
                .data
                .iter()
                .map(|point| point.y - (coefficient * point.x + intercept))
                .collect();

            self.residuals = Some(residuals);
        }
    }

    pub(crate) fn p_values_and_significance(&mut self) {
        let n = self.data.len() as f64;
        let k = 2.0; // Number of coefficients (including intercept and slope)
        let variance = self.residuals.clone().unwrap().iter().map(|res| res.powi(2)).sum::<f64>() / (n - k);
        let se_slope = (variance / (n * self.x_variance())).sqrt();
        let se_intercept = (variance * (1.0 / n + (self.x_mean().powi(2) / (n * self.x_variance())))).sqrt();

        // Two-tailed t-distribution test with alpha = 0.05 (95% confidence level)
        let t_critical = 2.0; // t-distribution critical value for alpha = 0.05 and two-tailed test
        let t_stat_slope = self.coefficient.expect("not calculated") / se_slope;
        let t_stat_intercept = self.intercept.expect("not calculated") / se_intercept;

        // Calculate p-values using statrs crate
        let t_dist = StudentsT::new(0.0, 1.0, n - k).unwrap();
        let p_value_slope = 2.0 * (1.0 - t_dist.cdf(t_stat_slope.abs()));
        let p_value_intercept = 2.0 * (1.0 - t_dist.cdf(t_stat_intercept.abs()));
        self.p_value_slope = Some(p_value_slope);
        self.p_value_intercept = Some(p_value_intercept);
    }

    fn x_variance(&self) -> f64 {
        let x_mean = self.x_mean();
        self.data.iter().map(|p| (p.clone().x - x_mean).powi(2)).sum::<f64>() / (self.data.len() as f64 - 1.0)
    }

    fn x_mean(&self) -> f64 {
        self.data.iter().map(|p| p.clone().x).sum::<f64>() / self.data.len() as f64
    }
    pub(crate) fn calc_confidence_level_and_intervals(&mut self) {
        if let (Some(coefficient), Some(residuals)) = (
            self.coefficient,
            &self.residuals
        ) {
            let n = self.data.len() as f64;
            let k = 2.0; // Number of coefficients (including intercept and slope)

            // Get the standard error for slope
            let variance = residuals.iter().map(|res| res.powi(2)).sum::<f64>() / (n - k);
            let se_slope = (variance / (n * self.x_variance())).sqrt();
            let t_stat = coefficient / se_slope;
            // Calculate t-distribution critical value
            let t_dist = StudentsT::new(0.0, 1.0, n - k).unwrap();
            self.confidence_level = Some(1.0 - (t_dist.cdf(t_stat.abs()) - t_dist.cdf(-t_stat.abs())));
            let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - self.confidence_level.unwrap()) / 2.0);

            // Calculate confidence intervals
            let slope_lower = coefficient - t_critical * se_slope;
            let slope_upper = coefficient + t_critical * se_slope;

            self.confidence_intervals = Some((slope_lower, slope_upper));

            // Calculate confidence level based on t-statistic
            let t_stat = coefficient / se_slope;

            // Print intermediate values for debugging
            println!("t_critical: {}", t_critical);
            println!("t_stat: {}", t_stat);

            // The confidence level should be calculated as the area under the t-distribution
            // curve outside the interval defined by the t-statistic (two-tailed test)
        } else {
            self.confidence_level = None;
            self.confidence_intervals = None;
        }
    }

        pub(crate) fn predict(self: &Self, x_values: &[f64]) -> Vec<DataPoint> {
            if let (Some(coefficient), Some(intercept)) = (self.coefficient, self.intercept) {
               let y_values: Vec<f64>=x_values
                    .iter()
                    .map(|&x| coefficient * x + intercept)
                    .collect();
              let mut  predictions : Vec<DataPoint> = vec![];
                for i in 0..x_values.len(){
                    predictions.push(DataPoint{x:x_values[i],y:y_values[i]});
                }
                predictions
            } else {
                vec![]
            }
        }


}
