use rand::distributions::Uniform;
use rand::distributions::Distribution;
use rand::Rng;
use rand::thread_rng;


fn main() 
{
    let mut rng = rand::thread_rng();
    let u = rng.gen::<f64>();
    print!("{}", u);

}


