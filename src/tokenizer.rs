use anyhow::Result;
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainer};
use tokenizers::normalizers::{Sequence as NormalizerSequence, StripAccents, NFD};
use tokenizers::pre_tokenizers::{
    punctuation::Punctuation, sequence::Sequence as PreTokenizerSequence, whitespace::Whitespace,
    PreTokenizerWrapper,
};
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{AddedToken, Tokenizer, TokenizerBuilder};

#[derive(Clone)]
pub struct LLMTokenizer {
    tokenizer: Tokenizer,
}

impl LLMTokenizer {
    pub fn new(training_file: &str) -> Result<Self> {
        let normalizer = NormalizerSequence::new(vec![NFD.into(), StripAccents.into()]);
        let decoder = WordPieceDecoder::default();

        let pre_tokenizer = PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Whitespace(Whitespace::default()),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ]);

        let mut tokenizer = TokenizerBuilder::new()
            .with_model(WordPiece::default())
            .with_normalizer(Some(normalizer))
            .with_pre_tokenizer(Some(pre_tokenizer))
            .with_decoder(Some(decoder))
            .with_post_processor(Some(BertProcessing::new(
                ("[SEP]".to_string(), 102),
                ("[CLS]".to_string(), 101),
            )))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build tokenizer: {}", e))?;

        let mut trainer = WordPieceTrainer::builder()
            .vocab_size(30000)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from("[UNK]", true),
                AddedToken::from("[SEP]", true),
                AddedToken::from("[PAD]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[MASK]", true),
            ])
            .continuing_subword_prefix("##".to_string())
            .build();

        let files = vec![training_file.to_string()];
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
