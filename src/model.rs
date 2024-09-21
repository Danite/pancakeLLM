use anyhow::Result;
use tch::nn::{Module, ModuleT};
use tch::{nn, Tensor};

#[derive(Debug, Clone)]
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
    pub config: LLMConfig, // Make the config field public
    token_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    encoder: Vec<TransformerLayer>,
    lm_head: nn::Linear,
}

#[derive(Debug)]
struct TransformerLayer {
    self_attention: MultiHeadAttention,
    attention_layer_norm: nn::LayerNorm,
    feed_forward: nn::Sequential,
    ff_layer_norm: nn::LayerNorm,
}

#[derive(Debug)]
struct MultiHeadAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    num_heads: i64,
}

impl PancakeLLM {
    pub fn new(vs: &nn::Path, config: &LLMConfig) -> Result<Self> {
        let token_embeddings = nn::embedding(
            vs / "token_embeddings",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let position_embeddings = nn::embedding(
            vs / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm = nn::layer_norm(
            vs / "layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        let mut encoder = Vec::new();
        for i in 0..config.num_hidden_layers {
            encoder.push(TransformerLayer::new(vs / format!("layer_{}", i), config));
        }

        let lm_head = nn::linear(
            vs / "lm_head",
            config.hidden_size,
            config.vocab_size,
            Default::default(),
        );

        Ok(Self {
            config: config.clone(),
            token_embeddings,
            position_embeddings,
            layer_norm,
            encoder,
            lm_head,
        })
    }
}

impl TransformerLayer {
    fn new(vs: nn::Path, config: &LLMConfig) -> Self {
        let self_attention = MultiHeadAttention::new(&(vs.clone() / "self_attention"), config);
        let attention_layer_norm = nn::layer_norm(
            vs.clone() / "attention_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        let feed_forward = nn::seq()
            .add(nn::linear(
                vs.clone() / "ff_1",
                config.hidden_size,
                config.intermediate_size,
                Default::default(),
            ))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(
                vs.clone() / "ff_2",
                config.intermediate_size,
                config.hidden_size,
                Default::default(),
            ));

        let ff_layer_norm = nn::layer_norm(
            vs / "ff_layer_norm",
            vec![config.hidden_size],
            Default::default(),
        );

        Self {
            self_attention,
            attention_layer_norm,
            feed_forward,
            ff_layer_norm,
        }
    }
}

impl MultiHeadAttention {
    fn new(vs: &nn::Path, config: &LLMConfig) -> Self {
        let query = nn::linear(
            vs / "query",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let key = nn::linear(
            vs / "key",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let value = nn::linear(
            vs / "value",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let output = nn::linear(
            vs / "output",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );

        Self {
            query,
            key,
            value,
            output,
            num_heads: config.num_attention_heads,
        }
    }
}

impl ModuleT for PancakeLLM {
    fn forward_t(&self, input_ids: &Tensor, train: bool) -> Tensor {
        let (batch_size, seq_len) = input_ids.size2().unwrap();
        let position_ids = Tensor::arange(seq_len, (input_ids.kind(), input_ids.device()))
            .unsqueeze(0)
            .expand(&[batch_size, seq_len], true);

        let token_embeds = self.token_embeddings.forward(input_ids);
        let position_embeds = self.position_embeddings.forward(&position_ids);

        let mut hidden_states = token_embeds + position_embeds;
        hidden_states = self.layer_norm.forward(&hidden_states);

        for layer in &self.encoder {
            hidden_states = layer.forward_t(&hidden_states, train);
        }

        self.lm_head.forward(&hidden_states)
    }
}

impl ModuleT for TransformerLayer {
    fn forward_t(&self, hidden_states: &Tensor, _train: bool) -> Tensor {
        let attention_output = self.self_attention.forward_t(hidden_states, _train);
        let hidden_states = self
            .attention_layer_norm
            .forward(&(hidden_states + attention_output));

        let ff_output = self.feed_forward.forward(&hidden_states);
        self.ff_layer_norm.forward(&(hidden_states + ff_output))
    }
}

impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, hidden_states: &Tensor, _train: bool) -> Tensor {
        let q = self.query.forward(hidden_states);
        let k = self.key.forward(hidden_states);
        let v = self.value.forward(hidden_states);

        let batch_size = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let head_dim = hidden_states.size()[2] / self.num_heads;

        let q = q
            .view((batch_size, seq_len, self.num_heads, head_dim))
            .transpose(1, 2);
        let k = k
            .view((batch_size, seq_len, self.num_heads, head_dim))
            .transpose(1, 2);
        let v = v
            .view((batch_size, seq_len, self.num_heads, head_dim))
            .transpose(1, 2);

        let attention_scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let attention_probs = attention_scores.softmax(-1, attention_scores.kind());

        let context = attention_probs
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view((batch_size, seq_len, -1));

        self.output.forward(&context)
    }
}
