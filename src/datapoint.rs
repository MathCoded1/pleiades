// Define a simple DataPoint struct
#[derive(Clone, Debug)]
pub struct DataPoint {
    x: f64,
    y: f64,
}
impl DataPoint{
    pub fn new(x:f64, y:f64) ->Self{
        DataPoint{x,y}
    }
    pub fn get_x(&self) ->f64{
        self.x
}
    pub fn get_y(&self) ->f64{
    self.y
}

}