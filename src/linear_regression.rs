use std::error::Error;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::style::full_palette::{RED, BLUE, WHITE};
use plotters::drawing::IntoDrawingArea;
use plotters::element::{Circle, EmptyElement};
use plotters::series::{LineSeries, PointSeries};


// Define a simple DataPoint struct
#[derive(Clone, Debug)]
pub(crate) struct DataPoint {
     x: f64,
     y: f64,
}
impl DataPoint{
    pub(crate) fn new(x:f64, y:f64) ->Self{
        DataPoint{x,y}
    }
}
#[derive(Debug)]
pub(crate) struct LinearRegression{
    data: Box<Vec<DataPoint>>,
    coefficient: f64,
    intercept: f64,
    goodness_of_fit:f64,
    residuals:f64,
    p_values_and_significance:f64,
    confidence_intervals:f64,
    predictions:f64,
    multicollinearity:f64,
    outliers:f64,
    homoscedasticity:f64,
}
impl LinearRegression {
    pub(crate) fn load(data : &Vec<DataPoint>) ->Self {
        LinearRegression {
            data: Box::new(data.to_vec()),
            coefficient: 0.0,
            intercept: 0.0,
            goodness_of_fit: 0.0,
            residuals: 0.0,
            p_values_and_significance: 0.0,
            confidence_intervals: 0.0,
            predictions: 0.0,
            multicollinearity: 0.0,
            outliers: 0.0,
            homoscedasticity: 0.0,
        }
    }

    pub(crate) fn calculate(&mut self){
        self.calc_slope_intercept().unwrap();
        self.calc_goodness_of_fit();

    }
    pub(crate) fn get_slope_intercept(&self) -> (f64,f64){
        (self.coefficient,self.intercept)
    }

    // Function to calculate the linear regression coefficients
        fn calc_slope_intercept(&mut self)->Result<(), Box<dyn Error>> {
        let n = self.data.len() as f64;
        let x_sum: f64 = self.data.iter().map(|p| p.x).sum();
        let y_sum: f64 = self.data.iter().map(|p| p.y).sum();
        let xy_sum: f64 = self.data.iter().map(|p| p.x * p.y).sum();
        let x_squared_sum: f64 =self.data.iter().map(|p| p.x * p.x).sum();

        self.coefficient = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum);
        self.intercept = (y_sum - self.coefficient * x_sum) / n;
        Ok(())
    }

// Function to plot the data and the linear regression line

    pub(crate) fn plot_data_with_regression(
        &self,
        slope: f64,
        intercept: f64,
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
                return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
            },
        ))?;

        // Draw the linear regression line
        chart.draw_series(LineSeries::new(
            vec![(0.0, intercept), (5.0, slope * 5.0 + intercept)],
            &RED,
        ))?;

        Ok(())
    }
    // Function to calculate the goodness of fit (R-squared)
    fn calc_goodness_of_fit(&mut self) {
        let y_mean: f64 = self.data.iter().map(|p| p.clone().y).sum::<f64>() / self.data.len() as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for point in self.data.iter() {
            let y_pred = self.coefficient * point.clone().x + self.intercept;
            let y_actual = point.clone().y;

            ss_res += (y_actual - y_pred).powi(2);
            ss_tot += (y_actual - y_mean).powi(2);
        }

        self.goodness_of_fit = 1.0 - ss_res / ss_tot;
    }

    // Function to get the R-squared value
    pub(crate) fn r_squared(&self) -> f64 {
        self.goodness_of_fit
    }
}


