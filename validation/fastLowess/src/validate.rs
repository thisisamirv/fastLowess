use fastLowess::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
struct ValidationData {
    name: String,
    notes: String,
    input: InputData,
    params: Params,
    #[serde(skip_deserializing)]
    result: ResultData,
}

#[derive(Debug, Deserialize, Serialize)]
struct InputData {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Params {
    fraction: f64,
    degree: Option<usize>, // Optional in R output now? No, implied 1.
    iterations: usize,
    delta: Option<f64>,
    #[serde(flatten)]
    extra: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResultData {
    fitted: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Relative to validation/fastLowess/
    let input_dir = Path::new("../output/r");
    let output_dir = Path::new("../output/fastLowess");

    if !input_dir.exists() {
        eprintln!(
            "Input directory {:?} does not exist. Run validate.R first.",
            input_dir
        );
        return Ok(());
    }

    fs::create_dir_all(output_dir)?;

    let mut entries: Vec<_> = fs::read_dir(input_dir)?.filter_map(|e| e.ok()).collect();

    // Sort to ensure deterministic processing order
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            println!("Processing {:?}", path.file_name().unwrap());
            process_file(&path, output_dir)?;
        }
    }

    Ok(())
}

fn process_file(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(input_path)?;
    let mut data: ValidationData = serde_json::from_reader(file)?;

    // Configure Lowess builder
    let mut builder = Lowess::new()
        .fraction(data.params.fraction)
        .iterations(data.params.iterations)
        .scaling_method(MAR)
        .boundary_policy(NoBoundary)
        .parallel(true)
        .adapter(Batch); // Explicitly use Batch adapter

    // Handle delta if present
    if let Some(d) = data.params.delta {
        builder = builder.delta(d);
    }

    let processor = builder.build()?;
    let result = processor.fit(&data.input.x, &data.input.y)?;

    data.result.fitted = result.y;

    let output_path = output_dir.join(input_path.file_name().unwrap());
    let output_json = serde_json::to_string_pretty(&data)?;
    fs::write(output_path, output_json)?;

    Ok(())
}
