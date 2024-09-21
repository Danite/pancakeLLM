use anyhow::Result;
use tch::{nn, Device, Tensor};

pub struct LLMConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
}

pub struct PancakeLLM {
    embeddings: nn::Embedding,
    encoder: nn::TransformerEncoder,
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

        let encoder_layer = nn::transformer_encoder_layer(
            vs / "encoder_layer",
            nn::TransformerEncoderLayerConfig {
                d_model: config.hidden_size,
                nhead: config.num_attention_heads,
                dim_feedforward: config.intermediate_size,
                ..Default::default()
            },
        );

        let encoder = nn::transformer_encoder(
            vs / "encoder",
            &encoder_layer,
            config.num_hidden_layers,
            None,
        );

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

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedded = self.embeddings.forward(input_ids);
        let encoded = self.encoder.forward(&embedded, None, None);
        let output = self.lm_head.forward(&encoded);
        Ok(output)
    }
}
