use std::fs;
fn main() {
    let common = fs::read_to_string("shaders/common.wgsl").unwrap();
    let opt = fs::read_to_string("shaders/optimize.wgsl").unwrap();
    let st = opt.lines().position(|l| l.starts_with("fn score_serial")).unwrap();
    let txt = common + "\n" + &opt.lines().skip(st).collect::<Vec<_>>().join("\n");
    match naga::front::wgsl::parse_str(&txt) {
        Ok(_) => println!("OK"),
        Err(e) => println!("{:?}", e),
    }
}
