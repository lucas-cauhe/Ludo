// Logistic Regression

// Para entrenar el modelo
// 1. Feature extraction
//      1.1. Preprocessing (Using tokenizers crates) 🦧
//          1.1.2 Eliminar stopwords, puntuación, urls, almohadillas...
//          1.1.3 Stemming y poner todo en minúsculas
//      1.2. Generar el diccionario
//          1.2.1 Diccionario de palabras
//          1.2.2 Diccionario de frecuencia de positivos y negativos
//      1.3 Crear los Xm y generar su matriz 🦧 
// 2. Entrenar el modelo 🦧
//      2.1 Implementar función sigmoid (o de predicción) h(X, theta)
//      2.2 Implementar función coste J(theta)
//      2.3 Implementar convergencia de theta (theta(j)-alpha*d(J(theta))/d(theta(j)))
//      2.3 Comprobar el resultado con la función coste

// Para realizar los test

// Naive-Bayes Approach
// Vector Spaces
extern crate rust_tokenizers;
extern crate rust_stemmers;
extern crate rand;
extern crate nalgebra;


mod lr;
mod tokenization;


use lr::main::LogisticRegression;
use nalgebra::{DMatrix};
use rust_stemmers::{Algorithm, Stemmer};


fn main(){
   
    //let tokens = tokenization().expect("Something went wrong");
    // save these tokens somehow
    let inputs = vec!["i love my brother", "i am doing well", "i am not doing fine", "i heard a scream last night so police came by", "Positive thinking will let you do everything better than negative thinking will", "It always seems impossible until it is done", "Keep your face to the sunshine and you cannot see a shadow", "They werent playing poker", "I am not flying to England"];
    let expected_outputs: Vec<i32> = [1, 1, 0, 0, 1, 1, 1, 0, 0].to_vec();
    let mut my_lr = LogisticRegression{
        inputs: inputs.to_vec(),
        dictionary_path: "".to_string(),
        frequency_arrays: None
    };
    
    let my_lr: LogisticRegression = LogisticRegression::input_model(&mut my_lr, &inputs.to_vec(), &"./vocabs/first_dict.json").expect("Something went wrong");
    
    let my_input_data: DMatrix<f64> = match &my_lr.frequency_arrays {
        Some(f) => 
            DMatrix::from_vec(3, f.len(), f.iter().flat_map(|k| {
                let fa: Vec<f64> = k.iter().map(|l| *l as f64).collect();
                
                fa
            })
            .collect()),
        None => DMatrix::from_vec(1, 1, vec![])
    };
    let trained_theta = my_lr.train_model(&my_input_data.transpose(), &expected_outputs);
    
    let predicted_data = my_lr.predict(&my_input_data.transpose(), &trained_theta);
    for t in &predicted_data{
        println!("{}", t);
    }
    

}
