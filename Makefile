BINARY  := target/release/primitive-gpu
CARGO   := cargo
LIB     := target/release/libprimitive_gpu.a
SOLIB   := target/release/libprimitive_gpu.so
CAPI    := --features capi
MONA    := /home/frans/src/primitive/examples/monalisa.png
GO_BIN  := /tmp/primitive-go
CC      ?= cc

.PHONY: all debug release lib clean test video image ab go example \
        capi capi-test filter help

all: release

debug:
	$(CARGO) build

release:
	$(CARGO) build --release

$(BINARY): src/main.rs src/lib.rs src/engine.rs src/shapes.rs src/io.rs src/gpuio.rs src/capi.rs shaders/*.wgsl Cargo.toml
	$(CARGO) build --release

lib: $(LIB)

# C ABI static + dynamic libraries
$(LIB) $(SOLIB): src/lib.rs src/capi.rs src/engine.rs src/shapes.rs src/io.rs src/gpuio.rs shaders/*.wgsl Cargo.toml
	$(CARGO) build --release $(CAPI)

capi: $(LIB) $(SOLIB)

capi-test: $(LIB)
	$(CC) -O2 -I c -o /tmp/test_capi c/test_capi.c $(LIB) -lpthread -ldl -lm
	/tmp/test_capi

# Native ffmpeg filter (requires libavfilter-dev, libavutil-dev)
libfilter_primitive.so: $(LIB) c/vf_primitive.c
	$(CC) -O2 -fPIC -shared \
	  -I c -I target \
	  $$(pkg-config --cflags libavfilter libavutil 2>/dev/null) \
	  c/vf_primitive.c $(LIB) \
	  $$(pkg-config --libs libavfilter libavutil 2>/dev/null) \
	  -lpthread -ldl -lm \
	  -o $@

filter: libfilter_primitive.so

test: release
	./$(BINARY) -i $(MONA) -o /tmp/pg_mona.png -n 60

image: release
	./$(BINARY) -i $(MONA) -o /tmp/pg_mona.png -n 200

# ffmpeg pipe smoke test: synthetic source -> effect -> h264
video: release
	ffmpeg -y -v error -f lavfi -i testsrc2=size=320x180:rate=10:duration=1 \
	    -pix_fmt rgb24 -f rawvideo - \
	  | ./$(BINARY) --video --vw 320 --vh 180 -n 5 \
	  | ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s 320x180 -r 10 -i - \
	    -c:v libx264 -preset fast -pix_fmt yuv420p /tmp/test_out.mp4
	@ffprobe -v error -count_frames -select_streams v:0 \
	    -show_entries stream=nb_read_frames -of csv=p=0 /tmp/test_out.mp4 \
	  && echo "video OK: 10 frames"

# Temporal-reuse video smoke test
video-reuse: release
	ffmpeg -y -v error -f lavfi -i testsrc2=size=320x180:rate=10:duration=3 \
	    -pix_fmt rgb24 -f rawvideo - \
	  | ./$(BINARY) --video --vw 320 --vh 180 -n 10 --reuse --max-shapes 100 \
	  | ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s 320x180 -r 10 -i - \
	    -c:v libx264 -preset fast -pix_fmt yuv420p /tmp/test_reuse.mp4
	@ffprobe -v error -count_frames -select_streams v:0 \
	    -show_entries stream=nb_read_frames -of csv=p=0 /tmp/test_reuse.mp4 \
	  && echo "video-reuse OK: 30 frames"

# A/B against the Go original
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
	$(CARGO) clean
	rm -f libfilter_primitive.so /tmp/test_capi

help:
	@echo "make          - build release binary"
	@echo "make debug    - build debug binary"
	@echo "make lib      - build Rust static+shared libs"
	@echo "make capi     - build C ABI libraries"
	@echo "make capi-test- build + run the C ABI smoke test"
	@echo "make filter   - build libfilter_primitive.so (needs libavfilter-dev)"
	@echo "make test     - run 60-shape monalisa smoke test"
	@echo "make image    - run 200-shape monalisa"
	@echo "make video    - ffmpeg pipe smoke test (synthetic source)"
	@echo "make video-reuse - temporal-reuse video smoke test (30 frames)"
	@echo "make go       - build the Go original for A/B"
	@echo "make ab       - benchmark Go vs GPU on monalisa"
	@echo "make clean"