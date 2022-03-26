
use std::{io, ops::Mul};
use std::path::Path;
//use std::io::prelude::*;
use std::io::Read;
use std::fs::File;
use serde_json::{ Value};
//use crate::tokenization::main::tokenization;
use nalgebra::{DVector, DMatrix};


pub struct LogisticRegression<'a> {
    pub inputs: Vec<&'a str>,
    pub dictionary_path: String,
    pub frequency_arrays: Option<Vec<Vec<i32>>>,
}


impl<'a> LogisticRegression<'a> {

    pub fn input_model(&mut self, inputs: &Vec<&'a str>, dict_path: &str) -> Result<LogisticRegression, io::Error> {
        // abrir el json y crear la tabla (Xm)
        let path = Path::new(dict_path);
        let display = path.display();
        
        let mut file: File = File::open(&path)?;
        let mut s = String::new();
        let freq_arrays = match file.read_to_string(&mut s) {
            Err(why) => panic!("couldn't read {}: {}", display, why),
            Ok(_) => self.process_file(&s)?,
        };
        

        
        Ok(self.init_model(inputs, dict_path, &freq_arrays))
    }

    fn init_model(&self, inputs: &Vec<&'a str>, dict: &str, freq_arrays: &Vec<Vec<i32>>) -> LogisticRegression {
        LogisticRegression {
            inputs: inputs.to_vec(),
            dictionary_path: dict.to_string(),
            frequency_arrays: Some(freq_arrays.to_vec()),
        }
    }

    fn process_file(&self, d: &str) -> serde_json::Result<Vec<Vec<i32>>> {
        let data: Value = serde_json::from_str(d)?;
        
        let freq_array: Vec<Vec<i32>> = self.inputs.iter()
            .map(|i| self.count_frequency(i.split(' ').collect(), data["wordlist"].as_array().unwrap(), &data["positive"].as_array().unwrap(), &data["negative"].as_array().unwrap()))
            .collect();
        
        Ok(freq_array)
    }

    fn count_frequency(&self, i: Vec<&str>, dict: &Vec<Value>, pos: &Vec<Value>, neg: &Vec<Value>) -> Vec<i32> {
        let mut positive_freq: i64 = 0;
        let mut negative_freq: i64 = 0;
        // Falta stemmizar cada l
        
        for l in i {
            
            match dict.iter().position(|w| &w.as_str().unwrap() == &l) {
                Some(p) => {
                    positive_freq += pos.get(p).unwrap().as_i64().unwrap();
                    negative_freq += neg.get(p).unwrap().as_i64().unwrap();
                },
                None => ()
            }
            
        }
        
        vec![1, positive_freq as i32, negative_freq as i32].to_vec()

    }

    fn sigmoid(&self, v: f64) -> f64 {
        if v < -40.0 {
            0.0
        } else if v > 40.0 {
            1.0
        } else {
            1.0 / (1.0 + f64::exp(-v))
        }
    }

    fn predict(&self, x: &DMatrix<f64>, theta: &DVector<f64>) -> DVector<f64> {

        let mut prod: DVector<f64> = x.mul(theta);
        for p in &mut prod {
            *p = self.sigmoid(*p);
        }
        prod.into_owned()
    }

    fn compute_cost(&self, outputs: &Vec<i32>, pred: &DVector<f64>) -> f64 {
        
        let mut cost: f64 = 0.0;
        
        for i in 0..outputs.len() {
            let y = f64::from(outputs[i]);
            let p = pred.get(i).unwrap();
            let case_positive = y*p.ln();
            let case_negative = (1.0-y)*(1.0-p).ln();
            cost +=  case_positive + case_negative;
            if cost.is_nan() {
                return f64::MAX
            }
            
        }
        -cost*(1.0/outputs.len() as f64)
    }

    fn derive_cost(&self, outputs: &Vec<i32>, pred: &DVector<f64>, i: &DVector<f64>) -> f64 {
        
        let mut derived: f64 = 0.0;
        for e in 0..outputs.len() {
            derived += pred.get(e).unwrap() - f64::from(outputs[e]);
            derived = derived*i.get(e).unwrap();
        }
        if derived.is_nan() {
            return f64::MAX
        }
        derived*(1.0/outputs.len() as f64)
    }
        

    pub fn train_model(&self, input_matrix: &DMatrix<f64>, expected_outputs: &Vec<i32>) -> DVector<f64>{
        // inicializar random theta
        use rand::prelude::*;
        
        let mut rng = rand::thread_rng();
        let init_theta: [f64; 3] = rng.gen();
        let mut theta: DVector<f64> = DVector::from_row_slice(&init_theta);
        for t in &theta{
            println!("{}", t);
        }
       
        // Defines el learning step
        let l_step = 0.02;
        let threshold = 20;
        let mut batches = 0;
        
        // gradient descent
        loop { 
            
            let prediction_results = self.predict(input_matrix, &theta);
            /* for i in &prediction_results {
                println!("{}", i);
            } */
            let cost: f64 = self.compute_cost(expected_outputs, &prediction_results);
            println!("Current cost: {}", cost);
            if self.compute_cost(expected_outputs, &theta) < 1.0 || batches >= threshold {
                print!("Batches performed: {}", batches);
                break;
            }
            
            for t in 0..3 {
                let input_params: DVector<f64> = DVector::from_column_slice(input_matrix.column(t).as_slice());
                let derived_cost = self.derive_cost(expected_outputs, &prediction_results, &input_params);
                theta[t] -= l_step*&derived_cost;

            }
            batches += 1;
        }
        theta
        
    }
}