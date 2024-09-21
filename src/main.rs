mod model;
mod tokenizer;
mod trainer;

use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::model::LLMConfig;
use crate::tokenizer::LLMTokenizer;
use crate::trainer::Trainer;

fn main() -> Result<()> {
    let config = LLMConfig {
        vocab_size: 30000,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 512,
    };

    let device = Device::cuda_if_available();
    let mut trainer = Trainer::new(&config, 1e-4, device)?;

    let _tokenizer = LLMTokenizer::new()?;

    // dummy dataset
    let dataset: Vec<Tensor> = (0..100)
        .map(|_| {
            let input_ids = Tensor::randint(config.vocab_size, &[50], (Kind::Int64, device));
            let labels = Tensor::randint(config.vocab_size, &[50], (Kind::Int64, device));
            Tensor::stack(&[input_ids, labels], 0)
        })
        .collect();

    trainer.train(&dataset, 10, 32)?;

    Ok(())
}
