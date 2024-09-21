use anyhow::Result;
use tch::nn::{Module, ModuleT};
use tch::{nn, Tensor};

pub struct LLMConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
}

#[derive(Debug)]
pub struct PancakeLLM {
    embeddings: nn::Embedding,
    encoder: Vec<nn::Sequential>,
    lm_head: nn::Linear,
}

impl PancakeLLM {
    pub fn new(vs: &nn::Path, config: &LLMConfig) -> Result<Self> {
        let embeddings = nn::embedding(
            vs / "embeddings",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let mut encoder = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = nn::seq()
                .add(nn::layer_norm(
                    vs / format!("layer_norm_{}", i),
                    vec![config.hidden_size],
                    Default::default(),
                ))
                .add(nn::linear(
                    vs / format!("attention_{}", i),
                    config.hidden_size,
                    config.hidden_size,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add(nn::linear(
                    vs / format!("ff_{}", i),
                    config.hidden_size,
                    config.intermediate_size,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add(nn::linear(
                    vs / format!("ff_out_{}", i),
                    config.intermediate_size,
                    config.hidden_size,
                    Default::default(),
                ));
            encoder.push(layer);
        }

        let lm_head = nn::linear(
            vs / "lm_head",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        Ok(Self {
            embeddings,
            encoder,
            lm_head,
        })
    }
}

impl ModuleT for PancakeLLM {
    fn forward_t(&self, input_ids: &Tensor, train: bool) -> Tensor {
        let mut embedded = self.embeddings.forward(input_ids);
        for layer in &self.encoder {
            embedded = layer.forward_t(&embedded, train);
        }
        self.lm_head.forward(&embedded)
    }
}
