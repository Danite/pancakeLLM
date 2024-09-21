use anyhow::Result;
use tch::{nn, Device, Tensor};

use crate::model::{LLMConfig, PancakeLLM};

pub struct Trainer {
    model: PancakeLLM,
    optimizer: nn::Optimizer,
    device: Device,
}

impl Trainer {
    pub fn new(config: &LLMConfig, learning_rate: f64, device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let model = PancakeLLM::new(&vs.root(), config)?;
        let optimizer = nn::Adam::default().build(&vs, learning_rate)?;

        Ok(Self {
            model,
            optimizer,
            device,
        })
    }

    pub fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor) -> Result<f64> {
        let loss = self.model.forward(input_ids)?.cross_entropy_loss(labels);

        self.optimizer.backward_step(&loss);

        Ok(f64::from(&loss))
    }

    pub fn train(&mut self, dataset: &[Tensor], num_epochs: usize, batch_size: i64) -> Result<()> {
        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let num_batches = dataset.len() as i64 / batch_size;

            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (batch_idx + 1) * batch_size;

                let batch =
                    Tensor::stack(&dataset[start as usize..end as usize], 0).to(self.device);

                let input_ids = batch.slice(1, 0, -1, 1);
                let labels = batch.slice(1, 1, batch.size()[1], 1);

                let loss = self.train_step(&input_ids, &labels)?;
                total_loss += loss;
            }

            println!(
                "Epoch {}: Average loss = {}",
                epoch + 1,
                total_loss / num_batches as f64
            );
        }

        Ok(())
    }
}
