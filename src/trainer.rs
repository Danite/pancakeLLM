use anyhow::Result;
use tch::nn::{ModuleT, OptimizerConfig};
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

    pub fn train_step(&mut self, batch: &Tensor) -> Result<f64> {
        let input_ids = batch.select(0, 0);
        let labels = batch.select(0, 1);

        let logits = self.model.forward_t(&input_ids, true);
        let loss =
            logits.cross_entropy_loss::<Tensor>(&labels, None, tch::Reduction::Mean, -100, 0.0);

        if loss.isnan().any().int64_value(&[]) != 0 {
            return Err(anyhow::anyhow!("NaN loss encountered"));
        }

        self.optimizer.backward_step(&loss);

        Ok(loss.double_value(&[]))
    }

    pub fn train(&mut self, dataset: &[Tensor], num_epochs: usize, _batch_size: i64) -> Result<()> {
        if dataset.is_empty() {
            return Err(anyhow::anyhow!("Dataset is empty"));
        }

        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let num_batches = dataset.len() as i64;

            for batch in dataset.iter() {
                match self.train_step(batch) {
                    Ok(loss) => total_loss += loss,
                    Err(e) => println!("Warning: Error in batch: {}", e),
                }
            }

            let avg_loss = if num_batches > 0 {
                total_loss / num_batches as f64
            } else {
                f64::NAN
            };

            println!("Epoch {}: Average loss = {:.6}", epoch + 1, avg_loss);
        }

        Ok(())
    }
}
