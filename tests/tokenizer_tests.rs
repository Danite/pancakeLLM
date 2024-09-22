use anyhow::Result;
use pancakellm::LLMTokenizer;
use std::fs::File;
use std::io::Write;
use unicode_categories::UnicodeCategories;

fn create_test_file(content: &str) -> Result<String> {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_dataset.jsonl");
    let mut file = File::create(&file_path)?;
    file.write_all(content.as_bytes())?;
    Ok(file_path.to_string_lossy().into_owned())
}

#[test]
fn test_tokenizer_creation() -> Result<()> {
    let test_content = r#"{"text": "Hello, world!"}
{"text": "This is a test."}
{"text": "UPPERCASE lowercase MixedCase"}
{"text": "Multiple    spaces and\ttabs"}
{"text": "Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?"}"#;
    let file_path = create_test_file(test_content)?;

    let tokenizer = LLMTokenizer::new(&file_path)?;
    assert!(tokenizer.get_vocab_size() > 0);

    Ok(())
}

#[test]
fn test_tokenizer_encode_decode() -> Result<()> {
    let test_content = r#"{"text": "Hello, world!"}
{"text": "This is a test."}
{"text": "UPPERCASE lowercase MixedCase"}
{"text": "Multiple    spaces and\ttabs"}
{"text": "Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?"}"#;
    let file_path = create_test_file(test_content)?;

    let tokenizer = LLMTokenizer::new(&file_path)?;

    let test_cases = vec![
        "Hello, world!",
        "This is a test.",
        "UPPERCASE lowercase MixedCase",
        "Multiple spaces and tabs",
        "Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?",
    ];

    for test_text in test_cases {
        let encoded = tokenizer.encode(test_text)?;
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded)?;
        assert_eq!(
            normalized_text(&decoded),
            normalized_text(test_text),
            "Failed to correctly encode and decode: {}",
            test_text
        );
    }

    Ok(())
}

#[test]
fn test_tokenizer_special_tokens() -> Result<()> {
    let test_content = r#"{"text": "Hello, world!"}
{"text": "This is a test."}"#;
    let file_path = create_test_file(test_content)?;

    let tokenizer = LLMTokenizer::new(&file_path)?;

    let special_tokens = vec!["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"];

    for token in special_tokens {
        let encoded = tokenizer.encode(token)?;
        assert_eq!(
            encoded.len(),
            1,
            "Special token {} should be encoded as a single token",
            token
        );

        let decoded = tokenizer.decode(&encoded)?;
        assert_eq!(
            decoded.trim(),
            token,
            "Decoded special token {} should match the original",
            token
        );
    }

    Ok(())
}

#[test]
fn test_tokenizer_whitespace_handling() -> Result<()> {
    let test_content = r#"{"text": "Hello,   world!"}
{"text": "This   is  a    test."}
{"text": "Multiple    spaces and\ttabs    here"}
{"text": "UPPERCASE lowercase MixedCase"}"#;
    let file_path = create_test_file(test_content)?;

    let tokenizer = LLMTokenizer::new(&file_path)?;

    let test_cases = vec![
        ("Hello,   world!", "Hello, world!"),
        ("This   is  a    test.", "This is a test."),
        (
            "Multiple    spaces and\ttabs    here",
            "Multiple spaces and tabs here",
        ),
        (
            "UPPERCASE lowercase MixedCase",
            "UPPERCASE lowercase MixedCase",
        ),
    ];

    for (input, expected) in test_cases {
        let encoded = tokenizer.encode(input)?;
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded)?;
        assert_eq!(
            decoded.trim(),
            expected,
            "Failed to handle whitespace correctly in: {}",
            input
        );
    }

    Ok(())
}

fn normalized_text(text: &str) -> String {
    text.chars()
        .filter(|&c| !c.is_whitespace() && !c.is_punctuation())
        .collect::<String>()
        .to_lowercase()
}
