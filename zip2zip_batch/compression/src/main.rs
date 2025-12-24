use clap::Parser;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use compression::{LZWCompressor, PaddingStrategy};

#[derive(Parser, Debug)]
#[command(name = "LZW Compressor")]
#[command(about = "Compress and decompress token IDs using LZW", long_about = None)]
struct Args {
    /// Pass token IDs directly via CLI: e.g., --input-ids 1 2 3 4
    #[arg(long, value_name = "ID", num_args = 1..)]
    input_ids: Option<Vec<usize>>,

    /// Path to input file containing token IDs (space/comma/newline separated)
    #[arg(short, long, value_name = "FILE")]
    input: Option<PathBuf>,

    /// Generate N random token IDs instead of using input file
    #[arg(long, value_name = "N")]
    random: Option<usize>,

    /// Vocabulary size (initial size of dictionary)
    #[arg(long, default_value = "10")]
    initial_vocab_size: usize,

    /// Maximum codebook (dictionary) size. If omitted, defaults to the input length.
    #[arg(long)]
    max_codebook_size: Option<usize>,

    /// Maximum number of subtokens per match
    #[arg(long, default_value = "4")]
    max_subtokens: usize,

    /// Pad token ID
    #[arg(long, default_value = "0")]
    pad_token_id: usize,

    /// Comma-separated list of token IDs to disable
    #[arg(long, value_delimiter = ',', value_name = "ID,...")]
    disabled_ids: Option<Vec<usize>>,

    /// Verbose output
    #[arg(long, default_value = "false")]
    verbose: bool,
}

fn parse_ids_from_file(path: &PathBuf) -> Vec<usize> {
    let contents = fs::read_to_string(path).expect("Failed to read input file");
    contents
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| usize::from_str(s).expect("Invalid token ID in input file"))
        .collect()
}

fn generate_random_ids(len: usize, vocab_size: usize) -> Vec<usize> {
    let rng = thread_rng();
    let dist = Uniform::new(0, vocab_size);
    rng.sample_iter(dist).take(len).collect()
}

fn main() {
    let args = Args::parse();

    let ids = match (&args.input_ids, &args.input, args.random) {
        (Some(id_list), _, _) => {
            println!("Using token IDs from CLI: {} tokens", id_list.len());
            id_list.clone()
        }
        (None, Some(path), _) => {
            println!("Reading token IDs from file: {:?}", path);
            parse_ids_from_file(path)
        }
        (None, None, Some(len)) => {
            println!("Generating {} random token IDs...", len);
            generate_random_ids(len, args.initial_vocab_size)
        }
        _ => {
            eprintln!(
                "Error: You must specify one of --input-ids, --input <file>, or --random <len>"
            );
            std::process::exit(1);
        }
    };

    let max_codebook_size = args.max_codebook_size.unwrap_or_else(|| ids.len());

    let compressor = LZWCompressor::new(
        args.initial_vocab_size,
        max_codebook_size,
        args.max_subtokens,
        args.pad_token_id,
        args.disabled_ids.clone(),
    );

    let start = Instant::now();
    let ((compressed_ids, codebook), i) =
        compressor.internal_encode(&ids, 0, PaddingStrategy::DoNotPad, false, None);
    let encode_time = start.elapsed();

    println!(
        "Encode time: {:?}, speed: {:.2}M tokens/s",
        encode_time,
        ids.len() as f64 / encode_time.as_secs_f64() / 1_000_000.0
    );

    let start = Instant::now();
    let decoded_ids = compressor.internaldecode(&compressed_ids);
    let decode_time = start.elapsed();

    println!(
        "Decode time: {:?}, speed: {:.2}M tokens/s",
        decode_time,
        ids.len() as f64 / decode_time.as_secs_f64() / 1_000_000.0
    );

    assert_eq!(
        ids.len(),
        decoded_ids.len(),
        "Decoded length does not match original"
    );

    for (i, (&orig, decoded)) in ids.iter().zip(decoded_ids.iter()).enumerate() {
        assert_eq!(
            orig, *decoded,
            "Mismatch at position {}: expected {}, got {}",
            i, orig, decoded
        );
    }

    // print compressed_ids
    println!("compressed_ids: {:?}", compressed_ids);

    // print codebook
    println!("codebook: {:?}", codebook.internal_codebook);

    // print i
    println!("i: {:?}", i);

    println!(
        "✔ Compression successful. Compression ratio: {:.2}x ({} → {})",
        ids.len() as f64 / compressed_ids.len() as f64,
        ids.len(),
        compressed_ids.len()
    );

    println!("✔ Compression and decompression succeeded.");
}
