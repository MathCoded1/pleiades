use std::error::Error;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::style::full_palette::{RED, BLUE, WHITE};
use plotters::drawing::IntoDrawingArea;
use plotters::element::{Circle, EmptyElement};
use plotters::series::{LineSeries, PointSeries};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, FisherSnedecor,StudentsT};
use crate::datapoint::DataPoint;

#[derive(Debug)]
pub(crate) struct LinearRegression {
    data: Box<Vec<DataPoint>>,
    coefficients: Option<DVector<f64>>,
    intercept: Option<f64>,
    goodness_of_fit: Option<f64>,
    residuals: Option<DVector<f64>>,
    p_values: Option<DVector<f64>>,
    t_critical: Option<f64>,
    confidence_level: Option<f64>,
    confidence_intervals: Option<DMatrix<f64>>,
    predictions: Option<DVector<f64>>,
    vif_values: Option<DVector<f64>>,
    durbin_watson_statistic: Option<f64>,
    outliers: Option<f64>,
    homoscedasticity: Option<f64>,
}

impl LinearRegression {
    pub(crate) fn new(data: &Vec<DataPoint>) -> Self {
        LinearRegression {
            data: Box::new(data.to_vec()),
            coefficients: None,
            intercept: None,
            goodness_of_fit: None,
            residuals: None,
            p_values: None,
            t_critical: None,
            confidence_level: Some(0.95),
            confidence_intervals: None,
            predictions: None,
            vif_values: None,
            durbin_watson_statistic: None,
            outliers: None,
            homoscedasticity: None,
        }
    }

    pub(crate) fn calculate(&mut self) -> Result<(), Box<dyn Error>> {
        self.calc_coefficients()?;
        self.calc_goodness_of_fit();
        self.calc_residuals();
        self.calc_p_values();
        self.calc_confidence_level_and_intervals();
        self.calc_multicollinearity();
        self.calc_homoscedasticity();
        self.detect_outliers();

        Ok(())
    }
    pub(crate) fn get_homoscedasticity(&self) -> f64{
        self.homoscedasticity.expect("Not Calculated")
    }
    pub(crate) fn get_durbin_watson_statistic(&self) -> f64{
        self.durbin_watson_statistic.expect("Not Calculated")
    }

    pub(crate) fn get_coefficients(&self) -> Option<&DVector<f64>> {
        self.coefficients.as_ref()
    }

    pub(crate) fn get_intercept(&self) -> Option<f64> {
        self.intercept
    }
    pub(crate) fn get_goodness_of_fit(&self) -> Option<f64> {
        self.goodness_of_fit
    }

    pub(crate) fn get_residuals(&self) -> Option<&DVector<f64>> {
        self.residuals.as_ref()
    }

    pub(crate) fn get_p_values(&self) -> Option<&DVector<f64>> {
        self.p_values.as_ref()
    }

    pub(crate) fn get_t_critical(&self) -> Option<f64> {
        self.t_critical
    }

    pub(crate) fn get_confidence_level(&self) -> Option<f64> {
        self.confidence_level
    }

    pub(crate) fn get_confidence_intervals(&self) -> Option<&DMatrix<f64>> {
        self.confidence_intervals.as_ref()
    }

    pub(crate) fn get_predictions(&self) -> Option<&DVector<f64>> {
        self.predictions.as_ref()
    }

    pub(crate) fn get_vif_values(&self) -> Option<&DVector<f64>> {
        self.vif_values.as_ref()
    }

    pub(crate) fn get_outliers(&self) -> Option<f64> {
        self.outliers
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
        if let Some(coefficients) = &self.coefficients {
            let x_min = self.data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let x_max = self.data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let y_min = coefficients[0] + coefficients[1] * x_min;
            let y_max = coefficients[0] + coefficients[1] * x_max;
            chart.draw_series(LineSeries::new(
                vec![(x_min, y_min), (x_max, y_max)],
                &RED,
            ))?;
        }

        Ok(())
    }

    // Function to calculate the goodness of fit (R-squared)
    fn calc_goodness_of_fit(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let y_mean: f64 = self.data.iter().map(|p| p.y).sum::<f64>() / self.data.len() as f64;
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;

            for point in self.data.iter() {
                let y_pred = coefficients[0] + coefficients[1] * point.x;
                let y_actual = point.y;

                ss_res += (y_actual - y_pred).powi(2);
                ss_tot += (y_actual - y_mean).powi(2);
            }

            self.goodness_of_fit = Some(1.0 - ss_res / ss_tot);
        }
    }

    // Function to calculate the residuals
    pub(crate) fn calc_residuals(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let residuals: Vec<f64> = self
                .data
                .iter()
                .map(|point| point.y - (coefficients[0] + coefficients[1] * point.x))
                .collect();

            self.residuals = Some(DVector::from_row_slice(&residuals));
        }
    }

    // Function to calculate the p-values of coefficients
    pub(crate) fn calc_p_values(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let n = self.data.len() as f64;
            let k = coefficients.len() as f64; // Number of coefficients (including intercept)

            let variance = self
                .residuals
                .clone()
                .unwrap()
                .iter()
                .map(|res| res.powi(2))
                .sum::<f64>()
                / (n - k);

            let x_matrix = self.get_x_matrix();

            let se_coefficients = (x_matrix.transpose() * x_matrix)
                .try_inverse()
                .map(|cov_matrix| {
                    let se_coeffs: Vec<f64> = cov_matrix.diagonal().iter().map(|var| (var * variance).sqrt()).collect();
                    DVector::from_row_slice(&se_coeffs)
                });

            if let Some(se_coeffs) = se_coefficients {
                let t_dist = StudentsT::new(0.0, 1.0, n - k).unwrap();
                let t_values: Vec<f64> = coefficients
                    .iter()
                    .zip(se_coeffs.iter())
                    .map(|(coeff, se)| coeff / se)
                    .collect();

                let p_values: Vec<f64> = t_values
                    .iter()
                    .map(|&t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
                    .collect();

                self.p_values = Some(DVector::from_row_slice(&p_values));
            }
        }
    }

    // Function to calculate the confidence level and confidence intervals of coefficients
    pub(crate) fn calc_confidence_level_and_intervals(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let n = self.data.len() as f64;
            let k = coefficients.len() as f64; // Number of coefficients (including intercept)

            let x_matrix = self.get_x_matrix();

            let variance = self
                .residuals
                .clone()
                .unwrap()
                .iter()
                .map(|res| res.powi(2))
                .sum::<f64>()
                / (n - k);

            let se_coefficients = (x_matrix.transpose() * x_matrix)
                .try_inverse()
                .map(|cov_matrix| {
                    let se_coeffs: Vec<f64> = cov_matrix.diagonal().iter().map(|var| (var * variance).sqrt()).collect();
                    DVector::from_row_slice(&se_coeffs)
                });

            if let Some(se_coeffs) = se_coefficients {
                let t_dist = StudentsT::new(0.0, 1.0, n - k).unwrap();
                let t_critical = t_dist.inverse_cdf(1.0 - (1.0 - self.confidence_level.unwrap()) / 2.0);

                let confidence_intervals: Vec<(f64, f64)> = coefficients
                    .iter()
                    .zip(se_coeffs.iter())
                    .map(|(coeff, se)| {
                        let lower = coeff - t_critical * se;
                        let upper = coeff + t_critical * se;
                        (lower, upper)
                    })
                    .collect();

                self.t_critical = Some(t_critical);
                self.confidence_intervals = Some(DMatrix::from_fn(2, coefficients.len(), |i, j| {
                    confidence_intervals[j].0
                }));
            }
        }
    }

    // Function to calculate the multicollinearity using VIF values
    pub(crate) fn calc_multicollinearity(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let x_matrix = self.get_x_matrix();

            let r_squared_values: Vec<f64> = (0..coefficients.len()).map(|i| {
                let y_values = x_matrix.column(i).clone_owned();
                let x_values = x_matrix.clone().remove_column(i);
                let y_values_clone = y_values.clone();
                let x_values_clone = x_values.clone();
                let new_coefficients = (x_values.transpose() * &x_values)
                    .try_inverse()
                    .and_then(move |inv| Some(inv * x_values.transpose() * &y_values))
                    .ok_or("Failed to calculate VIF value.")
                    .unwrap();
                let y_hat = x_values_clone * new_coefficients;
                let ss_residuals: f64 = y_values_clone
                    .iter()
                    .zip(y_hat.iter())
                    .map(|(actual, predicted)| (actual - predicted).powi(2))
                    .sum();
                let ss_total: f64 = y_values_clone.iter().map(|actual| (actual - y_values_clone.mean()).powi(2)).sum();
                1.0 - ss_residuals / ss_total
            }).collect();

            self.vif_values = Some(DVector::from_row_slice(&r_squared_values));
        }
    }

    // Function to calculate the homoscedasticity using the Breusch-Pagan test
    pub(crate) fn calc_homoscedasticity(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let n = self.data.len() as f64;
            let k = coefficients.len() as f64; // Number of coefficients (including intercept)

            let y_values: DVector<f64> = DVector::from_row_slice(&self.data.iter().map(|p| p.y).collect::<Vec<f64>>());
            let y_hat = self.get_x_matrix() * coefficients;
            let residuals = y_values - y_hat;

            let squared_residuals = residuals.iter().map(|res| res.powi(2));
            let residuals_sum: f64 = squared_residuals.clone().sum();
            let squared_residuals_sum: f64 = squared_residuals.clone().map(|res| res.powi(2)).sum();
            let squared_residuals_squared_sum: f64 = squared_residuals.clone().map(|res| res.powi(3)).sum();
            let squared_residuals_cubed_sum: f64 = squared_residuals.clone().map(|res| res.powi(4)).sum();

            let homoscedasticity_statistic = ((n / 2.0) * squared_residuals_squared_sum / squared_residuals_sum.powi(2))
                + ((n / 2.0) * squared_residuals_cubed_sum / squared_residuals_sum.powi(3));

            let f_dist = FisherSnedecor::new(2.0, n - k - 1.0).unwrap();
            let homoscedasticity_critical_value = f_dist.inverse_cdf(1.0 - self.confidence_level.unwrap() / 2.0);

            self.homoscedasticity = Some(homoscedasticity_statistic);
            self.outliers = Some(homoscedasticity_critical_value);
        }
    }

    // Function to detect outliers using the Durbin-Watson test
    pub(crate) fn detect_outliers(&mut self) {
        if let Some(coefficients) = &self.coefficients {
            let residuals = self.residuals.clone().unwrap();

            let durbin_watson_statistic = residuals.iter().fold(0.0, |acc, &res| {
                let pair = residuals.view(((res - 1.0) as usize, 0), (2, 1) );
                acc + (pair[(1, 0)] - pair[(0, 0)]).powi(2)
            }) / residuals.iter().map(|&res| res.powi(2)).sum::<f64>();

            self.durbin_watson_statistic = Some(durbin_watson_statistic);
        }
    }

    // Function to calculate the coefficients of the linear regression model
    pub(crate) fn calc_coefficients(&mut self) -> Result<(), Box<dyn Error>> {

        let x_matrix = self.get_x_matrix();

        let y_vector = DVector::from_column_slice(
            &self.data.iter().map(|p| p.y).collect::<Vec<f64>>(),
        );
        let x_matrix_clone=x_matrix.transpose().clone();
        // Calculate the linear regression coefficients using least squares
        let coefficients = (x_matrix.transpose() * x_matrix)
            .try_inverse()
            .ok_or_else(move || Box::<dyn Error>::from("Failed to calculate matrix inverse"))? * x_matrix_clone * y_vector;

        self.coefficients = Some(coefficients);
        self.intercept = Some(self.coefficients.clone().unwrap()[0]);
        Ok(())
    }


    fn get_x_matrix(&self) -> DMatrix<f64> {
        let n = self.data.len() as usize;
        let mut x_matrix = DMatrix::zeros(n, 2);

        for (i, point) in self.data.iter().enumerate() {
            x_matrix[(i, 0)] = 1.0;
            x_matrix[(i, 1)] = point.x;
        }

        x_matrix
    }

    pub(crate) fn predict(&self, x_values: &[f64]) -> Vec<DataPoint> {
        if let Some(coefficients) = &self.coefficients {
            let x_matrix = DMatrix::from_fn(x_values.len(), 2, |i, j| match j {
                0 => 1.0,                     // Intercept column
                1 => x_values[i],             // X-values column
                _ => unreachable!(),
            });

            let y_values = x_matrix * coefficients;

            let mut predictions: Vec<DataPoint> = vec![];
            for (i, &x) in x_values.iter().enumerate() {
                predictions.push(DataPoint {
                    x,
                    y: y_values[(i, 0)],
                });
            }

            predictions
        } else {
            vec![]
        }
    }
     ad
}

