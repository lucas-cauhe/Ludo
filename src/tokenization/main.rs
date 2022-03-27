
use std::io;
use rust_tokenizers::adapters::Example;
use rust_tokenizers::tokenizer::{ Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{ Vocab, BpePairVocab};

use rust_tokenizers::tokenizer::{Gpt2Tokenizer};
use rust_tokenizers::vocab::{Gpt2Vocab};

use rust_stemmers::{Algorithm, Stemmer};

use std::io::Read;
use std::fs::File;
use serde_json::{ Value };
use std::path::Path;

pub fn tokenization(input: &str) -> Result<Vec<i64>, io::Error> {
    let vocab_path = "../vocabs/gpt2-vocab.json";
    let merges = BpePairVocab::from_file("./vocabs/merges/gpt2-merge.txt").unwrap();

    let vocab = Gpt2Vocab::from_file(&vocab_path).unwrap();

    let original_sentence: &str = &input.to_lowercase();
    // Stemming

    // Create a stemmer for the english language
    let en_stemmer = Stemmer::create(Algorithm::English);

    let splitted_sentence: Vec<&str> = original_sentence.split(' ').collect();
    let mut processed_sentence: String = "".to_string();
    for w in splitted_sentence {
        processed_sentence.push_str(&en_stemmer.stem(w));
        processed_sentence.push(' ');
    }

    let test_sentence = Example::new_from_string(&processed_sentence);
    let gpt2_tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);

    let tokens = gpt2_tokenizer.encode(&test_sentence.sentence_1, None, 128, &TruncationStrategy::LongestFirst, 0);
    let tokens_vec = &tokens.token_ids;
    
    /* let decoded = gpt2_tokenizer.decode(tokens_vec, true, true);
    println!("{:?}", decoded); */
    
    Ok(tokens_vec.to_vec())
}



/* pub fn update_dict(){
    let path = Path::new("./vocabs/first_dict.json");

    let mut file: File = File::open(&path).unwrap();
    let mut s = String::new();

    let json_file = match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read: {}", why),
        Ok(_) => {
            let data: Value = serde_json::from_str(&s).unwrap();
            data
        }
    };
    let mut wordlist: &mut Vec<Value> = json_file["wordlist"].as_array().unwrap().as_mut();
    let mut positive_freq: &mut Vec<Value> = json_file["positive"].as_array().unwrap();
    let mut negative_freq = json_file["negative"].as_array().unwrap();
    let positive_sent = json_file["phrases"].as_object().unwrap().get("positives").unwrap().as_array().unwrap();
    let negative_sent = json_file["phrases"].as_object().unwrap().get("negatives").unwrap().as_array().unwrap();


    for sentence in positive_sent {
        let original_sentence = &sentence.as_str().unwrap().to_lowercase();
        // Stemming

        // Create a stemmer for the english language
        let en_stemmer = Stemmer::create(Algorithm::English);

        let splitted_sentence: Vec<&str> = original_sentence.split(' ').collect();
        
        for mut w in splitted_sentence {
            let w = &en_stemmer.stem(w);
            match wordlist.iter().position(|x| &x.as_str().unwrap()==&w) {
                Some(i) => {
                    println!("yey");
                },
                None => {
                    wordlist.push(Value::String(w.to_string()));
                    let ww: Value = serde_json::from_str("{\"val\": 1}").unwrap();
                    let mut one: Vec<Value> = vec![ww["val"]];
                    positive_freq.append(&mut one);
                    ()
                }
            }
        }
    }

} */