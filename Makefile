BINARY := target/release/primitive-gpu
CARGO  := cargo
MONA   := /home/frans/src/primitive/examples/monalisa.png
GO_BIN := /tmp/primitive-go

.PHONY: all debug release clean test video image ab go run-video example help

all: release

debug:
	cargo build

release:
	cargo build --release

$(BINARY): src/main.rs src/engine.rs src/shapes.rs src/io.rs src/gpuio.rs shaders/*.wgsl
	cargo build --release

test: release
	./$(BINARY) -i $(MONA) -o /tmp/pg_mona.png -n 60

image: release
	./$(BINARY) -i $(MONA) -o /tmp/pg_mona.png -n 200

# ffmpeg smoke test: synthetic source -> effect -> h264
video: release
	ffmpeg -y -v error -f lavfi -i testsrc2=size=320x180:rate=10:duration=1 \
	    -pix_fmt rgb24 -f rawvideo - \
	  | ./$(BINARY) --video --vw 320 --vh 180 -n 5 \
	  | ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s 320x180 -r 10 -i - \
	    -c:v libx264 -preset fast -pix_fmt yuv420p /tmp/test_out.mp4
	@ffprobe -v error -count_frames -select_streams v:0 \
	    -show_entries stream=nb_read_frames -of csv=p=0 /tmp/test_out.mp4 \
	  && echo "video OK: 10 frames"

# A/B against the Go original (builds it on first use; see README)
go:
	cd /home/frans/src/primitive && \
	  (test -f go.mod || go mod init github.com/fogleman/primitive) && \
	  go build -o $(GO_BIN) .

ab: release go
	@echo "--- Go ---"
	@bash -c 'time $(GO_BIN) -i $(MONA) -o /tmp/go_mona.png -n 200' 2>&1 | grep -E "real|user|sys"
	@echo "--- GPU ---"
	@bash -c 'time ./$(BINARY) -i $(MONA) -o /tmp/pg200.png -n 200' 2>&1 | grep -E "real|user|sys"

clean:
	cargo clean

help:
	@echo "make          - build release binary"
	@echo "make debug    - build debug binary"
	@echo "make test     - run 60-shape monalisa smoke test"
	@echo "make image    - run 200-shape monalisa"
	@echo "make video    - ffmpeg pipe smoke test (synthetic source)"
	@echo "make go       - build the Go original for A/B"
	@echo "make ab       - benchmark Go vs GPU on monalisa"
	@echo "make clean"