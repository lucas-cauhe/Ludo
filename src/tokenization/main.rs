
use std::io;
use rust_tokenizers::adapters::Example;
use rust_tokenizers::tokenizer::{ Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{ Vocab, BpePairVocab};

use rust_tokenizers::tokenizer::{Gpt2Tokenizer};
use rust_tokenizers::vocab::{Gpt2Vocab};

use rust_stemmers::{Algorithm, Stemmer};

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