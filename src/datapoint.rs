// Define a simple DataPoint struct
#[derive(Clone, Debug)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}
impl DataPoint{
    pub fn new(x:f64, y:f64) ->Self{
        DataPoint{x,y}
    }

}