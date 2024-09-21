use anyhow::Result;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tch::{Device, Tensor};

use crate::tokenizer::LLMTokenizer;

#[derive(Debug, Deserialize)]
struct DataItem {
    text: String,
}

pub struct Dataset {
    items: Vec<DataItem>,
    tokenizer: LLMTokenizer,
    max_length: usize,
    device: Device,
}

impl Dataset {
    pub fn new(
        file_path: &str,
        tokenizer: LLMTokenizer,
        max_length: usize,
        device: Device,
    ) -> Result<Self> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "Dataset file not found: {}. Current directory: {:?}",
                file_path,
                std::env::current_dir()?
            ));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();

        for line in reader.lines() {
            let item: DataItem = serde_json::from_str(&line?)?;
            items.push(item);
        }

        println!("Loaded {} items from the dataset", items.len());

        Ok(Self {
            items,
            tokenizer,
            max_length,
            device,
        })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get_batch(&self, batch_size: usize) -> Result<Tensor> {
        let actual_batch_size = std::cmp::min(batch_size, self.items.len());
        let mut batch_tokens = Vec::new();

        for item in self.items.iter().take(actual_batch_size) {
            let tokens = self.tokenizer.encode(&item.text)?;
            let padded_tokens = self.pad_tokens(tokens);
            batch_tokens.push(padded_tokens);
        }

        // Pad the batch if necessary
        while batch_tokens.len() < batch_size {
            batch_tokens.push(vec![0; self.max_length]);
        }

        // Convert Vec<Vec<u32>> to Vec<i64>
        let flat_batch: Vec<i64> = batch_tokens.iter().flatten().map(|&x| x as i64).collect();

        // Create a tensor from the flattened data
        let batch_tensor = Tensor::f_from_slice(&flat_batch)?
            .view([batch_size as i64, self.max_length as i64])
            .to_device(self.device);

        Ok(batch_tensor)
    }

    fn pad_tokens(&self, mut tokens: Vec<u32>) -> Vec<u32> {
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
        } else {
            tokens.resize(self.max_length, 0);
        }
        tokens
    }
}
