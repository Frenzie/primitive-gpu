#[path = "../shapes.rs"]
pub mod shapes;
use std::fs;
use std::process::exit;
fn parse(tag: &str, txt: &str) -> bool {
    match naga::front::wgsl::parse_str(txt) {
        Ok(_) => true,
        Err(e) => {
            println!("{tag}: {e}");
            false
        }
    }
}
fn main() {
    let common = fs::read_to_string("shaders/common.wgsl").unwrap();
    let _ = &common;
    let opt = fs::read_to_string("shaders/optimize.wgsl").unwrap();
    let full = opt.clone();
    parse("FULL", &full);
    // split full into top-level blocks and parse each with prelude
    let owned: Vec<String> = full.lines().map(|s| s.to_string()).collect();
    let lines: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            l.starts_with("fn ") || l.starts_with("@compute") || l.starts_with("struct ")
                || l.starts_with("var<")
        })
        .map(|(i, _)| i)
        .collect();
    for j in 0..starts.len() {
        let st = starts[j];
        let en = if j + 1 < starts.len() { starts[j + 1] } else { lines.len() };
        let block = lines[st..en].join("\n");
                parse(&format!("block@{st}"), &format!("{}\n{}", common, block_escape(&block)));
    }
}
fn block_escape(b: &str) -> String { b.to_string() }
