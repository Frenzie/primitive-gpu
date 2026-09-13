pub fn read_bytes(device: &wgpu::Device, buf: &wgpu::Buffer) -> Vec<u8> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();
    let data = slice.get_mapped_range().to_vec();
    buf.unmap();
    data
}