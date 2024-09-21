use anyhow::Result;
use tokenizers::decoders::wordpiece::WordPiece;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Lowercase, Sequence, StripAccents, NFD};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{AddedToken, Tokenizer, TokenizerBuilder};

#[derive(Clone)]
pub struct LLMTokenizer {
    tokenizer: Tokenizer,
}

impl LLMTokenizer {
    pub fn new() -> Result<Self> {
        let normalizer = Sequence::new(vec![NFD.into(), Lowercase.into(), StripAccents.into()]);
        let decoder = WordPiece::default();

        let mut tokenizer = TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(Some(normalizer))
            .with_pre_tokenizer(Some(Whitespace::default()))
            .with_decoder(Some(decoder))
            .with_post_processor(Some(BertProcessing::new(
                ("[SEP]".to_string(), 102),
                ("[CLS]".to_string(), 101),
            )))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build tokenizer: {}", e))?;

        // Train the tokenizer on the test dataset
        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(30000)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from("[UNK]", true),
                AddedToken::from("[SEP]", true),
                AddedToken::from("[PAD]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[MASK]", true),
            ])
            .build();

        let files = vec![String::from("data/test_dataset.jsonl")];
        tokenizer
            .train_from_files(&mut trainer, files)
            .map_err(|e| anyhow::anyhow!("Failed to train tokenizer: {}", e))?;

        println!(
            "Tokenizer vocabulary size: {}",
            tokenizer.get_vocab_size(true)
        );

        Ok(Self {
            tokenizer: tokenizer.into(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        println!("Tokenizer encoded text '{}' to {:?}", text, ids);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let decoded = self
            .tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode: {}", e))?;
        Ok(decoded)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}
