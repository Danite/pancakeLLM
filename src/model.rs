use anyhow::Result;
use rand::Rng;
use tch::nn::{Module, ModuleT};
use tch::{nn, Device, Kind, Tensor};

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

    pub fn generate(&self, input_ids: &Tensor, max_length: usize) -> Result<Vec<i64>> {
        println!("Generate input tensor size: {:?}", input_ids.size());
        if input_ids.size()[0] == 0 || input_ids.size()[1] == 0 {
            return Err(anyhow::anyhow!(
                "Empty input tensor. Size: {:?}",
                input_ids.size()
            ));
        }

        let mut current_ids = input_ids.shallow_clone();
        let mut generated_ids = Vec::new();

        for _ in 0..max_length {
            let logits = self.forward_t(&current_ids, false);
            let next_token_logits = logits.select(1, -1);
            let probs = next_token_logits.softmax(-1, Kind::Float);

            println!("Top 5 probabilities: {:?}", probs.topk(5, -1, true, true));

            // Sample from the probability distribution
            let next_token = self.sample_from_probs(&probs)?;

            generated_ids.push(next_token);

            // Append the new token to the current_ids
            current_ids = Tensor::cat(
                &[
                    current_ids,
                    Tensor::f_from_slice(&[next_token])?.view((1, 1)),
                ],
                1,
            );

            // Stop if we generate an EOS token
            if next_token == 2 {
                break;
            }
        }

        Ok(generated_ids)
    }

    fn sample_from_probs(&self, probs: &Tensor) -> Result<i64> {
        let probs_vec: Vec<f32> = probs.view(-1).try_into()?;
        let mut rng = rand::thread_rng();
        let mut cumsum = 0.0;
        let sample = rng.gen::<f32>();

        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                return Ok(i as i64);
            }
        }

        Ok((probs_vec.len() - 1) as i64) // Fallback to the last token if sampling fails
    }

    pub fn device(&self) -> Device {
        self.token_embeddings.ws.device()
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
    fn forward_t(&self, input_ids: &Tensor, _train: bool) -> Tensor {
        let token_embeddings = self.token_embeddings.forward(input_ids);
        let position_ids = Tensor::arange(input_ids.size()[1], (Kind::Int64, input_ids.device()));
        let position_embeddings = self.position_embeddings.forward(&position_ids);

        let mut hidden_states = token_embeddings + position_embeddings;
        hidden_states = self.layer_norm.forward(&hidden_states);

        for layer in &self.encoder {
            hidden_states = layer.forward(&hidden_states);
        }

        self.lm_head.forward(&hidden_states)
    }
}

impl Module for TransformerLayer {
    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let attention_output = self.self_attention.forward_t(hidden_states, false);
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
