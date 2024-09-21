mod dataset;
mod model;
mod tokenizer;
mod trainer;

use anyhow::Result;
use tch::Device;

use crate::dataset::Dataset;
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

    let tokenizer = LLMTokenizer::new()?;
    let dataset = Dataset::new(
        "data/test_dataset.jsonl",
        tokenizer,
        config.max_position_embeddings as usize,
        device,
    )?;

    trainer.train(&dataset, 10, 5)?;

    Ok(())
}
