use anyhow::Result;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{Lowercase, StripAccents, NFD};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{AddedToken, Tokenizer};

pub struct LLMTokenizer {
    tokenizer: Tokenizer,
}

impl LLMTokenizer {
    pub fn new() -> Result<Self> {
        let mut tokenizer = Tokenizer::new(BPE::default());

        // Set up normalizers
        tokenizer.with_normalizer(Some(NFD));
        tokenizer.with_normalizer(Some(Lowercase));
        tokenizer.with_normalizer(Some(StripAccents));

        // Set up pre-tokenizer
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        // Set up post-processor
        let bert_processing =
            BertProcessing::new(("[SEP]".to_string(), 102), ("[CLS]".to_string(), 101));
        tokenizer.with_post_processor(Some(bert_processing));

        // Add special tokens
        tokenizer.add_special_tokens(&[
            AddedToken::from("[UNK]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[MASK]", true),
        ]);

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let decoded = self.tokenizer.decode(ids, false).unwrap();
        Ok(decoded)
    }
}
