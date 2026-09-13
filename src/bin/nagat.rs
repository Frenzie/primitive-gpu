
fn main() {
    let t = "// c\nfn __probe() {}";
    match naga::front::wgsl::parse_str(t) {
        Ok(_) => println!("ok"),
        Err(e) => println!("err: {e}"),
    }
}
