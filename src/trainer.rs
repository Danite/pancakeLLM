use anyhow::Result;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Tensor};

use crate::dataset::Dataset;
use crate::model::{LLMConfig, PancakeLLM};

pub struct Trainer {
    pub model: PancakeLLM,
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
        let input_ids = batch.shallow_clone();
        let labels = batch.shallow_clone();

        println!("Input tensor shape: {:?}", input_ids.size());

        let logits = self.model.forward_t(&input_ids, true);
        println!("Logits shape: {:?}", logits.size());

        let loss = logits
            .view([-1, self.model.config.vocab_size])
            .cross_entropy_loss::<Tensor>(&labels.view(-1), None, tch::Reduction::Mean, -100, 0.0);

        if loss.isnan().any().int64_value(&[]) != 0 {
            return Err(anyhow::anyhow!("NaN loss encountered"));
        }

        self.optimizer.backward_step(&loss);

        Ok(loss.double_value(&[]))
    }

    pub fn train<'a>(
        &mut self,
        dataset: &Dataset<'a>,
        num_epochs: usize,
        batch_size: usize,
    ) -> Result<()> {
        for epoch in 0..num_epochs {
            let mut total_loss = 0.0;
            let mut num_batches = 0;

            for _ in 0..(dataset.len() + batch_size - 1) / batch_size {
                match dataset.get_batch(batch_size)? {
                    Some(batch) => match self.train_step(&batch) {
                        Ok(loss) => {
                            total_loss += loss;
                            num_batches += 1;
                        }
                        Err(e) => println!("Warning: Error in batch: {}", e),
                    },
                    None => break,
                }
            }

            let avg_loss = if num_batches > 0 {
                total_loss / num_batches as f64
            } else {
                0.0
            };
            println!("Epoch {}: Average loss = {:.6}", epoch + 1, avg_loss);
        }

        Ok(())
    }
}
