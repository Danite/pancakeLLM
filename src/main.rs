mod dataset;
mod model;
mod tokenizer;
mod trainer;

use anyhow::Result;
use std::sync::Arc;
use tch::{Device, Tensor};

use crate::dataset::Dataset;
use crate::model::{LLMConfig, PancakeLLM};
use crate::tokenizer::LLMTokenizer;
use crate::trainer::Trainer;

fn test_prediction(
    model: &PancakeLLM,
    tokenizer: &LLMTokenizer,
    input_text: &str,
    max_length: usize,
) -> Result<()> {
    let input_ids = tokenizer.encode(input_text)?;
    println!("Encoded input_ids: {:?}", input_ids);

    let input_ids: Vec<i64> = input_ids.into_iter().map(|x| x as i64).collect();
    println!("Converted input_ids: {:?}", input_ids);

    let input_tensor = Tensor::f_from_slice(&input_ids)?.view((1, -1));
    println!("Input tensor size: {:?}", input_tensor.size());

    let generated_ids = model.generate(&input_tensor, max_length)?;
    let generated_text = tokenizer.decode(
        &generated_ids
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<u32>>(),
    )?;

    println!("Input: {}", input_text);
    println!("Generated: {}", generated_text);

    Ok(())
}

fn main() -> Result<()> {
    let training_file = "data/test_dataset.jsonl";
    let tokenizer = Arc::new(
        LLMTokenizer::new(training_file)
            .map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))?,
    );

    let config = LLMConfig {
        vocab_size: tokenizer.get_vocab_size() as i64,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 512,
    };

    let device = Device::cuda_if_available();
    let mut trainer = Trainer::new(&config, 1e-4, device)?;

    let dataset = Dataset::new(
        "data/test_dataset.jsonl",
        &tokenizer,
        config.max_position_embeddings as usize,
        device,
    )?;

    trainer.train(&dataset, 50, 32)?;

    // Test prediction
    let test_input = "Hello, how are";
    test_prediction(&trainer.model, &tokenizer, test_input, 20)?;

    Ok(())
}
